from ultralytics import YOLO
import argparse
import os
import shutil
import yaml
import sys
from pathlib import Path

# --- ARGUMENTOS ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--patience', type=int, default=15)
opt = parser.parse_args()

def get_project_root():
    """Retorna o diretório raiz do projeto (onde estão os data_*.yaml)"""
    return Path(__file__).parent.parent.parent

def fix_data_yaml():
    """
    Lê o data_7k.yaml e cria uma cópia temporária com o 
    CAMINHO ABSOLUTO do dataset para evitar erros de 'File Not Found'.
    """
    project_root = get_project_root()
    data_yaml_path = project_root / "data_7k.yaml"
    
    if not data_yaml_path.exists():
        print(f"ERRO CRÍTICO: {data_yaml_path} não encontrado!")
        sys.exit(1)

    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Converte o path relativo para absoluto
    dataset_rel_path = data.get('path', 'datasets/dataset7k')
    abs_dataset_path = str(project_root / dataset_rel_path)
    
    # Força o caminho absoluto no YAML
    data['path'] = abs_dataset_path 
    
    # Salva um novo arquivo temporário específico para este treino
    temp_yaml = project_root / f"data_abs_{opt.dataset}.yaml"
    with open(temp_yaml, 'w') as f:
        yaml.dump(data, f)
    
    print(f"--> YAML temporário criado: {temp_yaml}")
    print(f"    Dataset Path fixado em: {abs_dataset_path}")
    
    return str(temp_yaml)

def train():
    print(f"--- Iniciando Treino YOLOv11m (Dataset: {opt.dataset}) ---")
    
    # 1. Prepara o YAML com caminho absoluto
    data_file = fix_data_yaml()

    # 2. Carrega o Modelo
    try:
        model = YOLO('yolo11m.pt')
    except Exception as e:
        print("Tentando baixar yolo11m.pt...")
        model = YOLO('yolo11m.pt') # O Ultralytics tenta baixar auto

    # 3. Configura Balanceamento (Class Weights)
    # Aumentar peso da classificação para compensar desbalanceamento (Frasco vs Danificado)
    cls_gain = 2.0 if opt.dataset == '3k' else 1.5
    
    # 4. Define Argumentos de Treino
    args = {
        'data': data_file,      # Usa o arquivo YAML corrigido
        'project': 'runs_temp',
        'name': f'yolo11_{opt.dataset}',
        'epochs': opt.epochs,
        'batch': opt.batch,
        'imgsz': 640,
        'optimizer': 'SGD',     # Paridade com Faster R-CNN (Fair Comparison)
        'patience': opt.patience,
        'workers': opt.workers,
        'cls': cls_gain,        # Balanceamento de Classes
        
        # Augmentation (Alinhado com o Paper/TCC)
        'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
        'degrees': 15.0, 'translate': 0.1, 'scale': 0.5, 
        'mosaic': 1.0, 'erasing': 0.4,
        
        # Configurações Gerais
        'device': 0,            # Usa a primeira GPU
        'exist_ok': True,       # Sobrescreve se necessário
        'plots': True,
        'save': True,
        'val': True,
        'verbose': True
    }
    
    # 5. Executa o Treino
    model.train(**args)
    
    # 6. Organização Final (Move arquivos para as pastas limpas)
    print("\n--> Organizando arquivos finais...")
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    
    # Caminhos de origem
    src_base = f"runs_temp/yolo11_{opt.dataset}"
    src_csv = f"{src_base}/results.csv"
    src_pt = f"{src_base}/weights/best.pt"
    src_cm = f"{src_base}/confusion_matrix.png"

    # Caminhos de destino
    dst_csv = f"metrics/yolo11_{opt.dataset}_results.csv"
    dst_pt = f"weights/yolo11_{opt.dataset}_best.pt"
    dst_cm = f"metrics/yolo11_{opt.dataset}_confusion_matrix.png"

    # Cópias seguras
    if os.path.exists(src_csv):
        shutil.copy(src_csv, dst_csv)
        print(f"    CSV Salvo: {dst_csv}")
    
    if os.path.exists(src_pt):
        shutil.copy(src_pt, dst_pt)
        print(f"    Pesos Salvos: {dst_pt}")

    if os.path.exists(src_cm):
        shutil.copy(src_cm, dst_cm)
        print(f"    Matriz Salva: {dst_cm}")

    # Remove o YAML temporário para não sujar
    if os.path.exists(data_file):
        os.remove(data_file)

if __name__ == '__main__':
    train()