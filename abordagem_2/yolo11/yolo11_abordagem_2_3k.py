from ultralytics import YOLO
from pathlib import Path

# --- PATHS ---
# Diretório raiz do projeto (sobe 3 níveis: yolo11 -> abordagem 2 -> root)
ROOT = Path(__file__).parent.parent.parent

# Arquivos de configuração
DATASET_YAML = ROOT / "data_3k.yaml"
CONFIG_YAML = ROOT / "abordagem_2" / "config.yaml"

# Diretório de saída
OUTPUT_DIR = ROOT / "abordagem_2" / "yolo11" / "runs"
PROJECT_NAME = "yolo11_abordagem_2_3k"

# --- TREINAMENTO ---
print(f"=== YOLO11 - Abordagem 2 - Dataset 3k ===")
print(f"Dataset: {DATASET_YAML}")
print(f"Config: {CONFIG_YAML}")

# Carrega o modelo (download automático se necessário)
model = YOLO('yolo11x.pt')

# Treina usando as configurações do config.yaml
results = model.train(
    data=str(DATASET_YAML),
    cfg=str(CONFIG_YAML),
    project=str(OUTPUT_DIR),
    name=PROJECT_NAME
)

print(f"Treinamento finalizado! Resultados salvos em: {OUTPUT_DIR}/{PROJECT_NAME}")
