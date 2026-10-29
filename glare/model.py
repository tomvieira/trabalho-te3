"""GLARE model: genomic LM backbone + optional LoRA + multi-task head."""

import inspect
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers.modeling_outputs import BaseModelOutputWithNoAttention


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Monkey-patch PEFT 0.18.x / PyTorch 2.7 incompatibility
# peft.tuners.tuners_utils._get_in_out_features() checks
#   isinstance(module.weight, torch.distributed.tensor.DTensor)
# but torch.distributed.tensor does not exist in all PyTorch builds.
# Patching the flag to False avoids the AttributeError.
# ---------------------------------------------------------------------------
try:
    import peft.tuners.tuners_utils as _peft_tuners_utils
    if getattr(_peft_tuners_utils, "_torch_supports_distributed", False):
        import torch.distributed
        if not hasattr(torch.distributed, "tensor"):
            _peft_tuners_utils._torch_supports_distributed = False
except Exception:
    pass


# ---------------------------------------------------------------------------
# Attention pooling
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """Learned attention pooling over token positions.

    A single linear layer produces a scalar score per token position.
    Scores are softmax-normalized (with padding masked to -inf) and used
    as weights for a weighted sum of the hidden states.

    This allows the model to learn *which* positions are most informative
    for TE classification (e.g. LTR termini, TIR inverted repeats) rather
    than treating all positions equally as in mean pooling.

    Args:
        hidden_dim: Dimensionality of the input hidden states.
                    Defaults to 1536 (NTv3-650M); pass 512 for GPN.
    """

    def __init__(self, hidden_dim: int = 1536):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted pooled representation.

        Args:
            hidden_states: ``(B, L, D)`` — last hidden state from backbone.
            attention_mask: ``(B, L)`` — 1 for real tokens, 0 for padding.

        Returns:
            ``(B, D)`` pooled representation.
        """
        scores = self.attn(hidden_states).squeeze(-1)           # (B, L)
        scores = scores.masked_fill(attention_mask == 0, -1e9)  # mask padding
        weights = torch.softmax(scores, dim=-1)                 # (B, L)
        return (hidden_states * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

class GLAREHead(nn.Module):
    """Multi-task classification head for Order (auxiliary) and Superfamily (primary) prediction.

    The order head provides a fast, stable hierarchical gradient signal that
    helps the backbone learn a useful TE-order representation early in training.
    The superfamily head is the primary classification target.

    At inference time, ``pred_order`` is derived from ``pred_superfamily`` via
    :func:`~glare.constants.sf_to_order` (splitting on ``"/"``), so the order
    head output is only used during training for regularisation.

    Architecture:
        shared trunk: Linear(input_dim → hidden_dim) → GELU → Dropout
        order_head:       Linear(hidden_dim → num_orders)        [auxiliary]
        superfamily_head: Linear(hidden_dim → num_superfamilies) [primary]
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 512,
        num_orders: int = 7,
        num_superfamilies: int = 45,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # 1536 → 512
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 512 → 256
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.order_head = nn.Linear(hidden_dim // 2, num_orders)
        self.superfamily_head = nn.Linear(hidden_dim // 2, num_superfamilies)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Pooled backbone embeddings, shape ``(B, input_dim)``.

        Returns:
            ``(order_logits, superfamily_logits)`` — each ``(B, num_classes)``.
        """
        h = self.trunk(x)
        return self.order_head(h), self.superfamily_head(h)


class CaduceusMaskedLMBackbone(nn.Module):
    """Feature-backbone adapter for Caduceus masked-LM checkpoints.

    GLARE needs sequence hidden states, while the requested Caduceus checkpoint
    is loaded as ``AutoModelForMaskedLM`` and normally returns MLM logits.  This
    adapter asks the masked-LM model for hidden states and re-emits a standard
    ``BaseModelOutputWithNoAttention`` so GLARE pooling never consumes logits as
    features.  It is used only if the loaded masked-LM object does not expose an
    underlying ``.caduceus`` backbone directly.
    """

    def __init__(self, masked_lm: nn.Module):
        super().__init__()
        self.masked_lm = masked_lm
        self.config = masked_lm.config
        self._set_caduceus_hidden_size(self.config)

    @staticmethod
    def _set_caduceus_hidden_size(config) -> None:
        d_model = getattr(config, "d_model", None)
        if d_model is None:
            return
        hidden_size = int(d_model) * (2 if getattr(config, "rcps", False) else 1)
        config.hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = True,
        return_dict: bool | None = True,
    ) -> BaseModelOutputWithNoAttention:
        outputs = self.masked_lm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "Caduceus masked-LM output did not include hidden_states even "
                "though output_hidden_states=True was requested."
            )
        last_hidden = hidden_states[-1]
        if return_dict is False:
            return (last_hidden, hidden_states) if output_hidden_states else last_hidden
        return BaseModelOutputWithNoAttention(
            last_hidden_state=last_hidden,
            hidden_states=hidden_states if output_hidden_states else None,
        )


# ---------------------------------------------------------------------------
# Full GLARE model
# ---------------------------------------------------------------------------

class GLAREModel(nn.Module):
    """Full GLARE model: genomic LM backbone + :class:`GLAREHead`.

    Supports multiple backbone families:

    * **NTv3-650M** (``InstaDeepAI/NTv3_650M_pre``): Transformer U-Net,
      hidden size 1536, loaded via a ``sys.path`` / ``importlib`` workaround
      required by transformers 4.44.x.
    * **GPN** (``songlab/gpn-brassicales``): purely convolutional, hidden
      size 512, loaded by importing ``gpn.model`` first to register the
      architecture with HuggingFace's AutoModel registry.
     * **DNABERT-2** (``zhihan1996/DNABERT-2-117M``): BPE transformer, loaded
       through its HuggingFace dynamic module directly to avoid AutoModel
       remote-code config-class registration mismatches in transformers 4.44.x.
     * **Caduceus** (for example
       ``kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16``):
       masked-LM checkpoint loaded with ``AutoModelForMaskedLM`` and adapted to
       expose sequence hidden states for GLARE pooling.

    The backbone type is detected automatically from *base_model_name*
    (case-insensitive substrings such as ``"gpn"`` or ``"dnabert-2"``).  After loading, the actual
    hidden size is read from ``backbone.config.hidden_size`` and passed
    to :class:`AttentionPool` and :class:`GLAREHead`.
    """

    def __init__(
        self,
        base_model_name: str,
        lora_config: LoraConfig | None,
        use_lora: bool = True,
        hidden_dim: int = 512,
        num_orders: int = 7,
        num_superfamilies: int = 45,
        pooling: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()

        # Load base backbone using robust multi-strategy loading
        backbone = self._load_backbone(base_model_name)

        # Auto-detect hidden size from backbone config; fall back to 1536 (NTv3).
        try:
            hidden_size = backbone.config.hidden_size
        except AttributeError:
            hidden_size = 1536

        self.use_lora = use_lora

        if self.use_lora:
            if lora_config is None:
                raise ValueError("lora_config is required when use_lora=True")
            # Wrap with LoRA. PEFT freezes the base backbone and leaves only
            # adapter weights trainable, preserving the original default mode.
            self.backbone = get_peft_model(backbone, lora_config)
        else:
            # Full fine-tuning: skip PEFT and ensure all backbone parameters are
            # trainable even if a loaded backbone changes its defaults.
            self.backbone = backbone
            for param in self.backbone.parameters():
                param.requires_grad = True
        self._logged_dropped_backbone_kwargs: set[tuple[str, ...]] = set()

        # Pooling strategy
        self.pooling_type = pooling
        if pooling == "attention":
            self.pool = AttentionPool(hidden_dim=hidden_size)

        # Classification head — input_dim driven by backbone's hidden size
        self.head = GLAREHead(
            input_dim=hidden_size,
            hidden_dim=hidden_dim,
            num_orders=num_orders,
            num_superfamilies=num_superfamilies,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Helper: find the cached ntv3_base_model directory
    # ------------------------------------------------------------------

    @staticmethod
    def _find_ntv3_module_dir():
        """Find the cached transformers_modules directory for ntv3_base_model.

        ``get_class_from_dynamic_module`` in transformers==4.44.x fails for
        NTv3 because its internal ``check_imports()`` scans the modeling file
        for imports and misidentifies the relative import
        ``from .configuration_ntv3_pretrained import ...`` as a missing pip
        package named ``configuration_ntv3_pretrained``.

        This helper locates the directory that contains
        ``modeling_ntv3_pretrained.py`` so we can add it to ``sys.path``
        and import directly — bypassing the broken check entirely.

        Returns:
            str path to the directory, or ``None`` if not found.
        """
        import os
        from pathlib import Path

        candidate_bases = []

        # 1. transformers internal TRANSFORMERS_CACHE constant
        # TRANSFORMERS_CACHE typically points to ~/.cache/huggingface/hub,
        # but transformers_modules lives in the *sibling* ~/.cache/huggingface/modules.
        # So we check both the path itself AND its parent as bases.
        try:
            from transformers.utils import TRANSFORMERS_CACHE
            if TRANSFORMERS_CACHE:
                tc = Path(TRANSFORMERS_CACHE)
                candidate_bases.append(tc)           # in case it points to the hf root
                candidate_bases.append(tc.parent)    # parent of 'hub' → huggingface root
        except ImportError:
            pass

        # 2. HF_HOME env var → <HF_HOME>/modules
        hf_home = os.environ.get("HF_HOME", "")
        if hf_home:
            candidate_bases.append(Path(hf_home) / "modules")
            candidate_bases.append(Path(hf_home))

        # 3. HUGGINGFACE_HUB_CACHE env var (usually points to the hub subdir)
        hf_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "")
        if hf_hub_cache:
            candidate_bases.append(Path(hf_hub_cache).parent)   # parent of hub dir
            candidate_bases.append(Path(hf_hub_cache) / "modules")

        # 4. HF_HUB_CACHE env var (alternate spelling)
        hf_hub_cache2 = os.environ.get("HF_HUB_CACHE", "")
        if hf_hub_cache2:
            candidate_bases.append(Path(hf_hub_cache2).parent)
            candidate_bases.append(Path(hf_hub_cache2) / "modules")

        # 5. Standard default: ~/.cache/huggingface (root) and modules subdir
        hf_default = Path.home() / ".cache" / "huggingface"
        candidate_bases.append(hf_default)
        candidate_bases.append(hf_default / "modules")

        for base in candidate_bases:
            # Try both base/transformers_modules AND base/modules/transformers_modules
            # because HF cache layout can be:
            #   <hf_root>/hub/           (TRANSFORMERS_CACHE)
            #   <hf_root>/modules/transformers_modules/   ← the real location
            for tm in [base / "transformers_modules", base / "modules" / "transformers_modules"]:
                if not tm.exists():
                    continue
                for candidate in tm.glob(
                    "InstaDeepAI/ntv3_base_model/*/modeling_ntv3_pretrained.py"
                ):
                    return str(candidate.parent)

        # Hardcoded fallback: direct glob search across all known cache roots
        import glob as _glob
        for root in [
            "/global/scratch/users/nhaas/.cache/huggingface",
            "/global/home/users/nhaas/.cache/huggingface",
        ]:
            matches = _glob.glob(
                f"{root}/modules/transformers_modules/InstaDeepAI/ntv3_base_model/*/modeling_ntv3_pretrained.py"
            )
            if matches:
                return str(Path(matches[0]).parent)

        return None

    # ------------------------------------------------------------------
    # Backbone loader
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dnabert2_backbone(base_model_name: str):
        """Load DNABERT-2 without triggering ``AutoModel`` registration checks.

        DNABERT-2 ships a remote ``configuration_bert.BertConfig`` class and a
        remote ``bert_layers.BertModel`` class whose ``config_class`` points at
        transformers' built-in ``BertConfig``.  In transformers 4.44.x,
        ``AutoModel.from_pretrained(..., trust_remote_code=True)`` tries to
        register that remote model class against the remote config class and
        raises because those class objects differ.

        Loading the remote model class explicitly and aligning its
        ``config_class`` to the already-loaded remote config bypasses the
        ``AutoModel`` registration path while preserving the model code and
        weights selected by the repository's ``auto_map``.

        DNABERT-2's cached remote ``bert_layers.py`` routes attention through a
        Triton flash-attention kernel whenever its module-level
        ``flash_attn_qkvpacked_func`` imported successfully and attention
        dropout is exactly zero.  Some cluster jobs expose CUDA to PyTorch/DDP
        but not to Triton's driver discovery, causing the first forward pass to
        fail with ``RuntimeError: 0 active drivers``.  The remote code already
        has a PyTorch attention fallback for this situation; disable only that
        module-level flash function before model construction so DNABERT-2 uses
        the fallback without changing tokenizer behavior, weights, or other
        backbone families.
        """
        import sys

        from transformers import AutoConfig, AutoModel
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
        auto_map = getattr(config, "auto_map", None) or {}
        auto_model_ref = auto_map.get("AutoModel")

        if not auto_model_ref:
            # Conservative fallback for non-standard DNABERT-2 mirrors that do
            # not advertise a dynamic AutoModel entry.
            return AutoModel.from_pretrained(
                base_model_name, trust_remote_code=True, config=config
            )

        model_class = get_class_from_dynamic_module(auto_model_ref, base_model_name)
        model_module = sys.modules.get(model_class.__module__)
        if (
            model_module is not None
            and getattr(model_module, "flash_attn_qkvpacked_func", None) is not None
        ):
            logger.info(
                "Disabling DNABERT-2 remote Triton flash attention; using the "
                "remote PyTorch attention fallback instead."
            )
            model_module.flash_attn_qkvpacked_func = None

        model_config_class = getattr(model_class, "config_class", None)
        if model_config_class is not config.__class__:
            logger.info(
                "Patching DNABERT-2 remote model config_class from %s to %s "
                "before direct from_pretrained load.",
                model_config_class,
                config.__class__,
            )
            model_class.config_class = config.__class__

        return model_class.from_pretrained(base_model_name, config=config)

    @staticmethod
    def _set_caduceus_hidden_size(config) -> None:
        """Populate ``config.hidden_size`` for Caduceus configs."""
        d_model = getattr(config, "d_model", None)
        if d_model is None:
            return
        config.hidden_size = int(d_model) * (2 if getattr(config, "rcps", False) else 1)

    @staticmethod
    def _load_caduceus_backbone(base_model_name: str):
        """Load Caduceus from its masked-LM checkpoint as a feature backbone."""
        from transformers import AutoModelForMaskedLM

        masked_lm = AutoModelForMaskedLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
        GLAREModel._set_caduceus_hidden_size(masked_lm.config)

        caduceus = getattr(masked_lm, "caduceus", None)
        if caduceus is not None:
            # Prefer the exposed base model so GLARE receives last hidden states
            # directly and does not carry the MLM head through training.
            caduceus.config = masked_lm.config
            return caduceus

        logger.info(
            "Caduceus masked-LM object did not expose a .caduceus backbone; "
            "wrapping the MLM to return hidden states for GLARE pooling."
        )
        return CaduceusMaskedLMBackbone(masked_lm)

    @staticmethod
    def _load_backbone(base_model_name: str):
        """Load backbone model, dispatching on backbone family.

        **PlantCAD2** (name contains ``"plantcad"``, case-insensitive):
            Loads via ``AutoModel.from_pretrained`` with ``trust_remote_code=True``.
            RCPS (Reverse Complement Parameter Sharing) doubles the channel dim,
            so ``backbone.config.hidden_size`` is set to ``d_model * 2``.

        **Caduceus** (name contains ``"caduceus"``, case-insensitive):
            Loads the masked-language-model checkpoint via
            ``AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)``
            and returns the exposed base backbone (or a hidden-state adapter).

        **DNABERT-2** (name contains ``"dnabert-2"`` or ``"dnabert2"``):
            Loads the repository's remote ``AutoModel`` class directly and
            aligns its ``config_class`` when necessary, avoiding transformers
            4.44.x ``AutoModel`` registration mismatches. Tokenization-specific
            nucleotide context handling lives in :class:`glare.data.DNABERT2Tokenizer`.

        **GPN** (name contains ``"gpn"``, case-insensitive):
            Imports ``gpn.model`` to register the architecture with
            HuggingFace's AutoModel registry, then calls
            ``AutoModel.from_pretrained`` without ``trust_remote_code``.

        **NTv3** (all other names):
            Uses the ``sys.path`` / ``importlib`` workaround required by
            transformers 4.44.x (see :meth:`_find_ntv3_module_dir`).
        """
        from transformers import AutoModel, AutoConfig

        base_model_path = Path(str(base_model_name))
        is_local_dir = base_model_path.is_dir()

        if is_local_dir:
            # Inference bundles point model.base_model at a sibling local
            # save_pretrained() directory. Try that directory directly before
            # falling back to family-specific Hub/cache workarounds. This is
            # especially important for NTv3 bundles, where relying only on the
            # Hugging Face dynamic-module cache would reintroduce a runtime
            # dependency on the gated upstream repository.
            try:
                return AutoModel.from_pretrained(base_model_path, trust_remote_code=True)
            except Exception as exc:
                logger.info(
                    "Direct local AutoModel load from %s failed; trying GLARE "
                    "family-specific loader fallback: %s",
                    base_model_path,
                    exc,
                )

        if "caduceus" in base_model_name.lower():
            return GLAREModel._load_caduceus_backbone(base_model_name)

        if "plantcad" in base_model_name.lower():
            # PlantCAD2: Mamba/SSM-based bidirectional model.
            # trust_remote_code=True is required; no special import needed.
            backbone = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
            # RCPS doubles the channel dim: effective hidden_size = d_model * 2
            hidden_multiplier = 2 if getattr(backbone.config, "rcps", True) else 1
            backbone.config.hidden_size = backbone.config.d_model * hidden_multiplier
            return backbone

        normalized_name = base_model_name.lower().replace("_", "-")
        if "dnabert-2" in normalized_name or "dnabert2" in normalized_name:
            return GLAREModel._load_dnabert2_backbone(base_model_name)

        if "gpn" in base_model_name.lower():
            # GPN: register model class via side-effect import, then load normally.
            import gpn.model  # noqa: F401 — registers GPN with AutoModel
            return AutoModel.from_pretrained(base_model_name)

        # --- NTv3 path ---
        import importlib

        # Loading the config caches the remote code into transformers_modules.
        config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)

        # Prefer local dynamic-module files from a save_pretrained() bundle if
        # they are present. Some transformers versions/custom models do not copy
        # remote Python files during save_pretrained(); in that case we still
        # fall back to the existing cache workaround and document the limitation
        # in the training-time manifest.
        local_ntv3_file = base_model_path / "modeling_ntv3_pretrained.py"
        if is_local_dir and local_ntv3_file.is_file():
            ntv3_dir = str(base_model_path)
        else:
            # Locate the cached ntv3_base_model directory.
            ntv3_dir = GLAREModel._find_ntv3_module_dir()
        if ntv3_dir is None:
            raise RuntimeError(
                "Could not find NTv3 cached module directory under any "
                "HuggingFace cache location. Ensure that "
                "AutoConfig.from_pretrained(..., trust_remote_code=True) "
                "was called first and completed successfully, or provide a "
                "local bundle directory that includes modeling_ntv3_pretrained.py."
            )

        # Add to sys.path so that sibling relative imports resolve.
        if ntv3_dir not in sys.path:
            sys.path.insert(0, ntv3_dir)

        # Remove any stale cached import so importlib reloads from the dir.
        if "modeling_ntv3_pretrained" in sys.modules:
            del sys.modules["modeling_ntv3_pretrained"]

        modeling_module = importlib.import_module("modeling_ntv3_pretrained")
        NTv3PreTrained = getattr(modeling_module, "NTv3PreTrained")

        return NTv3PreTrained.from_pretrained(
            base_model_name, trust_remote_code=True, config=config
        )

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def _get_backbone_forward_signature(self):
        """Return the most specific forward signature we can find for the backbone.

        PEFT wrappers often expose permissive ``forward(**kwargs)`` methods, which
        is not enough to tell whether the underlying remote-code backbone accepts
        HuggingFace-style arguments such as ``attention_mask``. This helper walks
        common wrapper attributes until it finds a concrete ``forward`` signature
        without a catch-all ``**kwargs``.
        """
        candidates = [self.backbone]
        seen = set()

        while candidates:
            module = candidates.pop(0)
            if module is None or id(module) in seen:
                continue
            seen.add(id(module))

            forward = getattr(module, "forward", None)
            if callable(forward):
                try:
                    signature = inspect.signature(forward)
                except (TypeError, ValueError):
                    signature = None
                if signature is not None:
                    has_var_kwargs = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in signature.parameters.values()
                    )
                    if not has_var_kwargs:
                        return signature

            get_base_model = getattr(module, "get_base_model", None)
            if callable(get_base_model):
                try:
                    candidates.append(get_base_model())
                except Exception:
                    pass

            for attr in ("base_model", "model"):
                nested = getattr(module, attr, None)
                if nested is not None:
                    candidates.append(nested)

        forward = getattr(self.backbone, "forward", None)
        if callable(forward):
            try:
                return inspect.signature(forward)
            except (TypeError, ValueError):
                return None
        return None

    def _filter_backbone_kwargs(self, **kwargs):
        """Drop unsupported backbone kwargs based on the resolved forward signature."""
        signature = self._get_backbone_forward_signature()
        if signature is None:
            return kwargs, ()

        accepted = set(signature.parameters)
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        dropped = tuple(sorted(k for k in kwargs if k not in accepted))
        if dropped and dropped not in self._logged_dropped_backbone_kwargs:
            logger.info(
                "DIAG Backbone forward does not accept kwargs %s; calling without them.",
                ", ".join(dropped),
            )
            self._logged_dropped_backbone_kwargs.add(dropped)
        return filtered, dropped

    def _call_backbone(self, **kwargs):
        """Call the backbone while preserving LoRA when PEFT wrapper kwargs are incompatible.

        ``PeftModelForFeatureExtraction.forward`` always re-injects
        ``attention_mask`` and other HuggingFace-style kwargs into its wrapped
        base model. For remote-code backbones such as Caduceus, that defeats any
        prior kwarg filtering. In that case, call the underlying LoRA-wrapped
        model directly via ``self.backbone.base_model.model``.
        """
        filtered_kwargs, dropped = self._filter_backbone_kwargs(**kwargs)
        if dropped and hasattr(self.backbone, "base_model") and hasattr(self.backbone.base_model, "model"):
            return self.backbone.base_model.model(**filtered_kwargs)
        return self.backbone(**filtered_kwargs)

    def mean_pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pool last hidden state over non-pad positions.

        Args:
            hidden_states: ``(B, L, D)`` — last hidden state from backbone.
            attention_mask: ``(B, L)`` — 1 for real tokens, 0 for padding.

        Returns:
            ``(B, D)`` pooled representation.
        """
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (hidden_states * mask).sum(dim=1)   # (B, D)
        count = mask.sum(dim=1).clamp(min=1)         # (B, 1)
        return summed / count                         # (B, D)

    # ------------------------------------------------------------------
    # Trunk embedding extraction (for OOD detection)
    # ------------------------------------------------------------------

    def extract_trunk_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run backbone + pooling + head trunk and return 256-dim embeddings.

        Stops **before** the SF and order linear heads so the returned tensor
        is the shared 256-dim representation used by both heads.  Intended for
        use with :class:`~glare.ood.MahalanobisOOD` fitting and scoring.

        Args:
            input_ids: ``(B, L)``
            attention_mask: ``(B, L)`` — optional; constructed from input_ids
                if not provided.

        Returns:
            ``(B, 256)`` trunk embeddings (after ``GLAREHead.trunk``).
        """
        _, _, trunk_embeddings = self._forward_impl(input_ids, attention_mask)
        return trunk_embeddings

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the shared backbone once and return logits plus trunk embeddings."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        backbone_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        outputs = self._call_backbone(**backbone_kwargs)

        if getattr(outputs, "hidden_states", None) is not None:
            last_hidden = outputs.hidden_states[-1]
        elif getattr(outputs, "last_hidden_state", None) is not None:
            last_hidden = outputs.last_hidden_state
        else:
            last_hidden = outputs[0]

        if self.pooling_type == "attention":
            pooled = self.pool(last_hidden, attention_mask)
        else:
            pooled = self.mean_pool(last_hidden, attention_mask)

        trunk_embeddings = self.head.trunk(pooled)
        order_logits = self.head.order_head(trunk_embeddings)
        superfamily_logits = self.head.superfamily_head(trunk_embeddings)
        return order_logits, superfamily_logits, trunk_embeddings

    def forward_with_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return order/SF logits and the matching trunk embedding in one pass.

        This inference-oriented method is opt-in so the long-standing
        :meth:`forward` two-tensor return contract remains unchanged.
        """
        return self._forward_impl(input_ids, attention_mask)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run backbone + pooling + classification head.

        Args:
            input_ids: ``(B, L)``
            attention_mask: ``(B, L)`` — optional; constructed from input_ids
                if not provided (NTv3Tokenizer does not return one).

        Returns:
            ``(order_logits, superfamily_logits)`` — each ``(B, num_classes)``.
        """
        order_logits, superfamily_logits, _ = self._forward_impl(
            input_ids, attention_mask
        )
        return order_logits, superfamily_logits


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_model(config: dict) -> GLAREModel:
    """Build :class:`GLAREModel` from a nested config dict."""
    use_lora = config["model"].get("use_lora", True)
    lora_config = None
    if use_lora:
        lora_config = LoraConfig(
            r=config["model"]["lora_r"],
            lora_alpha=config["model"]["lora_alpha"],
            lora_dropout=config["model"]["lora_dropout"],
            target_modules=config["model"]["lora_target_modules"],
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
    return GLAREModel(
        base_model_name=config["model"]["base_model"],
        lora_config=lora_config,
        use_lora=use_lora,
        hidden_dim=config["model"]["hidden_dim"],
        num_orders=config["model"]["num_orders"],
        num_superfamilies=config["model"]["num_superfamilies"],
        pooling=config["model"].get("pooling", "mean"),   # default "mean" for backward compat
        dropout=config["model"].get("head_dropout", 0.1),  # default 0.1 for backward compat
    )


def count_parameters(model: GLAREModel) -> tuple[int, int]:
    """Return ``(trainable_params, total_params)`` for *model*."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
