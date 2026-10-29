#!/bin/bash
set -e  # Para imediatamente se qualquer comando falhar

export CUDA_VISIBLE_DEVICES=0

echo "Iniciando o treinamento do GLARE com meu_dataset.yaml..."
glare train --config configs/meu_dataset.yaml --output-dir ./resultados_meu_dataset

echo "Iniciando avaliação no conjunto de teste..."
glare evaluate \
    --config configs/meu_dataset.yaml \
    --checkpoint ./resultados_meu_dataset/best_model.pt

echo "Processo finalizado!"