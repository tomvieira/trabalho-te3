#!/bin/bash

# ==============================================================================
# Script para treinar e avaliar o GLARE oficial com dados customizados
# ==============================================================================

# 1. Definir o dispositivo (força o uso da GPU 0)
export CUDA_VISIBLE_DEVICES=0

# 2. Iniciar o treinamento
echo "Iniciando o treinamento do GLARE com meu_dataset.yaml..."
glare train --config configs/meu_dataset.yaml --output-dir ./resultados_meu_dataset

# 3. Realizar a predição/avaliação no conjunto de teste
# O comando exato dependerá das opções do CLI (verifique rodando `glare --help`)
echo "Iniciando avaliação no conjunto de teste..."
glare evaluate \
    --config configs/meu_dataset.yaml \
    --checkpoint ./resultados_meu_dataset/best_model.pt \
    --metrics-out ./resultados_meu_dataset/metricas_finais.csv

echo "Processo finalizado!"