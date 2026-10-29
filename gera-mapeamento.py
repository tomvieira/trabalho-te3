import os

# Caminho para a sua pasta original de treino
TRAIN_DIR = './data/Dataset3/Train'
# Caminho onde o ficheiro TOML deve ser salvo (dentro do repositório GLARE)
TOML_PATH = './meu_mapeamento.toml'

def gerar_mapeamento():
    if not os.path.exists(TRAIN_DIR):
        print(f"Erro: A pasta {TRAIN_DIR} não existe.")
        return

    # Extrai os nomes das classes (nomes dos ficheiros sem extensão)
    arquivos = sorted(f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(('.fa', '.fasta', '.fna')))
    classes = [os.path.splitext(f)[0] for f in arquivos]

    if not classes:
        print("Erro: Nenhum ficheiro FASTA encontrado na pasta.")
        return

    # Cria a pasta configs se não existir
    os.makedirs(os.path.dirname(TOML_PATH), exist_ok=True)

    # Escreve o ficheiro TOML
    with open(TOML_PATH, 'w') as f:
        f.write("[Custom]\n")
        for c in classes:
            f.write(f'"{c}" = "{c}"\n')

    print(f"Sucesso! O ficheiro {TOML_PATH} foi gerado com as seguintes {len(classes)} classes:")
    for c in classes:
        print(f" - {c}")

if __name__ == "__main__":
    gerar_mapeamento()