"""
Baixa o dataset MIMIC-III (v1.4) do PhysioNet para data/raw/mimic/.

Requer uma conta credenciada no PhysioNet com acesso ao MIMIC-III (inclui
completar o treinamento CITI) — ver https://physionet.org/content/mimiciii/.
Edite `username` abaixo antes de rodar; a senha é pedida interativamente
(não fica visível em `ps`/histórico do shell).

Depois de baixar, rode `python data/merge_mimic_tables.py` para gerar o
`merged_data.csv` que `src/utils/hospital_splitter.py` espera.
"""

import getpass
import glob
import subprocess

username = "seu_email@example.com"


if __name__ == "__main__":
    if username == "seu_email@example.com":
        raise SystemExit(
            "Edite a variável `username` neste arquivo com seu e-mail do PhysioNet antes de rodar."
        )

    password = getpass.getpass("Senha PhysioNet: ")

    # -nH --cut-dirs=3 evita que o wget replique a árvore de diretórios do URL
    # (physionet.org/files/mimiciii/1.4/...) dentro de data/raw/mimic/.
    subprocess.run(
        [
            "wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3",
            "--user", username, "--password", password,
            "https://physionet.org/files/mimiciii/1.4/",
            "-P", "./data/raw/mimic/",
        ],
        check=True,
    )

    csv_files = glob.glob("data/raw/mimic/**/*.csv", recursive=True)
    print(f"✓ {len(csv_files)} arquivos CSV encontrados em data/raw/mimic/")
    print("Próximo passo: python data/merge_mimic_tables.py")
