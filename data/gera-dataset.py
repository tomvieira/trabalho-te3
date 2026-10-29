import os

# Defina os caminhos de origem e destino
IN_DIR = './Dataset3'
OUT_DIR = './Dataset3_nova'


def formatar_fasta_para_glare():
    for split in ['Train', 'Test']:
        in_split = os.path.join(IN_DIR, split)
        out_split = os.path.join(OUT_DIR, split)

        # Verifica se a pasta Train/Test existe
        if not os.path.exists(in_split):
            print(f"Aviso: Pasta {in_split} não encontrada. Ignorando.")
            continue

        # Cria a pasta de destino
        os.makedirs(out_split, exist_ok=True)
        print(f"Processando pasta: {in_split} -> {out_split}")

        # Itera sobre todos os ficheiros FASTA
        for filename in os.listdir(in_split):
            if filename.lower().endswith(('.fa', '.fasta', '.fna')):
                # Ex: 'LTR' extraído de 'LTR.fa'
                class_name = os.path.splitext(filename)[0]
                in_filepath = os.path.join(in_split, filename)
                out_filepath = os.path.join(out_split, filename)

                contagem = 0
                with open(in_filepath, 'r') as f_in, open(out_filepath, 'w') as f_out:
                    for line in f_in:
                        line = line.strip()
                        if line.startswith('>'):
                            # Pega o ID original ignorando o '>' e qualquer espaço depois
                            original_id = line[1:].split()[0]

                            # Formata para o padrão esperado pelo RepBase loader do GLARE
                            # Formato: >ID#Ordem/Superfamilia
                            novo_cabecalho = f">{original_id}#Custom/{class_name}\n"
                            f_out.write(novo_cabecalho)
                            contagem += 1
                        else:
                            if line:  # Ignora linhas em branco
                                f_out.write(f"{line}\n")

                print(
                    f"  - {filename}: {contagem} sequências formatadas com '#Custom/{class_name}'")


if __name__ == "__main__":
    formatar_fasta_para_glare()
    print("\nConcluído! A pasta Dataset3_nova está pronta para ser usada no GLARE.")
