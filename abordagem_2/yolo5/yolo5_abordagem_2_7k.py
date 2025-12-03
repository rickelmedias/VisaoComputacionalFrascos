from ultralytics import YOLO
from pathlib import Path

# --- PATHS ---
# Diretório raiz do projeto (sobe 3 níveis: yolo5 -> abordagem 2 -> root)
ROOT = Path(__file__).parent.parent.parent

# Arquivos de configuração
DATASET_YAML = ROOT / "data_7k.yaml"
CONFIG_YAML = ROOT / "abordagem_2" / "config.yaml"

# Modelo pré-treinado (será baixado automaticamente se não existir)
MODEL_NAME = "yolov5x6u.pt"

# --- TREINAMENTO ---
print(f"=== YOLOv5 - Abordagem 2 - Dataset 7k ===")
print(f"Dataset: {DATASET_YAML}")
print(f"Config: {CONFIG_YAML}")

model = YOLO(MODEL_NAME)
results = model.train(
    data=str(DATASET_YAML),
    cfg=str(CONFIG_YAML),
    project=str(ROOT / "abordagem_2" / "yolo5" / "runs"),
    name="yolo5_abordagem_2_7k"
)

print("Treinamento finalizado!")
