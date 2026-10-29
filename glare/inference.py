"""Shared record-level inference aggregation utilities."""

from __future__ import annotations

import torch


class MeanLogitAccumulator:
    """Accumulate float32 window outputs by dense source-record index.

    Both model heads are always aggregated as raw logits.  Optional embeddings
    use the same window counts, allowing Mahalanobis scoring on each record's
    mean window embedding.  :meth:`means` rejects incomplete inference rather
    than silently manufacturing outputs for records with no windows.
    """

    def __init__(self, n_records: int, device: torch.device | str = "cpu"):
        if n_records < 0:
            raise ValueError(f"n_records must be non-negative, got {n_records}.")
        self.n_records = n_records
        self.device = torch.device(device)
        self.counts = torch.zeros(n_records, dtype=torch.float32, device=self.device)
        self._order_sums: torch.Tensor | None = None
        self._superfamily_sums: torch.Tensor | None = None
        self._embedding_sums: torch.Tensor | None = None

    def _ensure_sum(self, current: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError(f"Aggregated tensors must be rank 2, got shape {tuple(values.shape)}.")
        if current is None:
            return torch.zeros(
                self.n_records,
                values.shape[1],
                dtype=torch.float32,
                device=self.device,
            )
        if current.shape[1] != values.shape[1]:
            raise ValueError(
                f"Inconsistent aggregated width: expected {current.shape[1]}, got {values.shape[1]}."
            )
        return current

    def add(
        self,
        record_indices: torch.Tensor,
        order_logits: torch.Tensor,
        superfamily_logits: torch.Tensor,
        embeddings: torch.Tensor | None = None,
    ) -> None:
        """Add one window batch after converting all values to float32."""
        indices = record_indices.detach().to(self.device, dtype=torch.long)
        if indices.ndim != 1:
            raise ValueError("record_indices must be a rank-1 tensor.")
        batch_size = indices.shape[0]
        if order_logits.shape[0] != batch_size or superfamily_logits.shape[0] != batch_size:
            raise ValueError("Logit batch sizes must match record_indices.")
        if batch_size and (int(indices.min()) < 0 or int(indices.max()) >= self.n_records):
            raise IndexError("record_indices contains an out-of-range source-record index.")

        order_values = order_logits.detach().to(self.device, dtype=torch.float32)
        sf_values = superfamily_logits.detach().to(self.device, dtype=torch.float32)
        self._order_sums = self._ensure_sum(self._order_sums, order_values)
        self._superfamily_sums = self._ensure_sum(self._superfamily_sums, sf_values)
        self._order_sums.index_add_(0, indices, order_values)
        self._superfamily_sums.index_add_(0, indices, sf_values)
        self.counts.index_add_(
            0,
            indices,
            torch.ones(batch_size, dtype=torch.float32, device=self.device),
        )

        if embeddings is not None:
            if embeddings.shape[0] != batch_size:
                raise ValueError("Embedding batch size must match record_indices.")
            embedding_values = embeddings.detach().to(self.device, dtype=torch.float32)
            self._embedding_sums = self._ensure_sum(self._embedding_sums, embedding_values)
            self._embedding_sums.index_add_(0, indices, embedding_values)
        elif self._embedding_sums is not None:
            raise ValueError("Embeddings must be supplied consistently for every batch.")

    def _validated_denominator(self) -> torch.Tensor:
        missing = torch.nonzero(self.counts == 0, as_tuple=False).flatten().tolist()
        if missing:
            preview = missing[:10]
            suffix = "..." if len(missing) > 10 else ""
            raise ValueError(f"Missing windows for source records {preview}{suffix}.")
        if self._order_sums is None or self._superfamily_sums is None:
            raise ValueError("No logits were accumulated.")
        return self.counts.unsqueeze(1)

    def means(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean raw order and superfamily logits in source order."""
        denominator = self._validated_denominator()
        assert self._order_sums is not None and self._superfamily_sums is not None
        return self._order_sums / denominator, self._superfamily_sums / denominator

    def mean_embeddings(self) -> torch.Tensor:
        """Return mean window embeddings in source order."""
        denominator = self._validated_denominator()
        if self._embedding_sums is None:
            raise ValueError("No embeddings were accumulated.")
        return self._embedding_sums / denominator
