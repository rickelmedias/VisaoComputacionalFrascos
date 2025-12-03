import yaml
import os
import sys
import argparse
import shutil
import subprocess
from pathlib import Path

# --- ARGUMENTOS ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--patience', type=int, default=15)
opt = parser.parse_args()

def create_hyperparameters():
    """Cria o arquivo de hiperparâmetros com base no dataset (3k/7k)"""
    # 1. Tenta carregar padrão para evitar KeyError
    default_hyp = "yolov5/data/hyps/hyp.scratch-low.yaml"
    if os.path.exists(default_hyp):
        with open(default_hyp, 'r') as f:
            hyp = yaml.safe_load(f)
    else:
        # Fallback
        hyp = {'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0, 'iou_t': 0.20, 'anchor_t': 4.0, 'fl_gamma': 0.0, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0}

    # 2. Nossos ajustes
    cls_pw = 2.0 if opt.dataset == '3k' else 1.5
    hyp.update({
        'cls': 0.5 * cls_pw, 
        'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
        'degrees': 15.0, 'translate': 0.1, 'scale': 0.5, 'shear': 10.0,
        'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0
    })
    
    fname = f'hyp_{opt.dataset}.yaml'
    with open(fname, 'w') as f: yaml.dump(hyp, f)
    return fname

def get_project_root():
    """Retorna o diretório raiz do projeto (onde estão os data_*.yaml)"""
    return Path(__file__).parent.parent.parent

def fix_data_yaml():
    """Lê o data_3k.yaml e cria uma cópia com CAMINHO ABSOLUTO"""
    project_root = get_project_root()
    data_yaml_path = project_root / "data_3k.yaml"
    
    if not data_yaml_path.exists():
        print(f"ERRO: {data_yaml_path} não encontrado!")
        sys.exit(1)

    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Converte o path relativo para absoluto
    dataset_rel_path = data.get('path', 'datasets/dataset3k')
    abs_dataset_path = str(project_root / dataset_rel_path)
    
    # Atualiza o path no yaml
    data['path'] = abs_dataset_path 
    
    # Salva um novo arquivo temporário
    temp_yaml = project_root / f"data_abs_{opt.dataset}.yaml"
    with open(temp_yaml, 'w') as f:
        yaml.dump(data, f)
    
    print(f"--> Criado YAML com caminho absoluto: {temp_yaml}")
    print(f"    Path definido como: {abs_dataset_path}")
    
    return str(temp_yaml)

def train():
    hyp_file = create_hyperparameters()
    
    # Gera o data.yaml com caminho absoluto para não confundir o YOLO
    data_file = fix_data_yaml()
    
    # Clone de segurança
    if not os.path.exists('yolov5'):
        print("Clonando YOLOv5...")
        os.system('git clone https://github.com/ultralytics/yolov5')
        
    print(f"--> Iniciando Treino YOLOv5 {opt.dataset}...")

    cmd = [
        sys.executable, "yolov5/train.py",
        "--img", "640",
        "--batch", str(opt.batch),
        "--epochs", str(opt.epochs),
        "--data", data_file,              # Usa o arquivo corrigido
        "--weights", "yolov5m.pt",
        "--project", "runs_temp",
        "--name", f"yolo5_{opt.dataset}",
        "--hyp", hyp_file,
        "--optimizer", "SGD",
        "--exist-ok",
        "--workers", str(opt.workers),
        "--patience", str(opt.patience)
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERRO NO TREINO YOLOv5: {e}")
        return

    # Organização final
    print("Organizando arquivos de saída...")
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    
    base = f"runs_temp/yolo5_{opt.dataset}"
    
    if os.path.exists(f"{base}/results.csv"):
        shutil.copy(f"{base}/results.csv", f"metrics/yolo5_{opt.dataset}_results.csv")
        print(f"Métricas salvas em metrics/yolo5_{opt.dataset}_results.csv")
    
    if os.path.exists(f"{base}/weights/best.pt"):
        shutil.copy(f"{base}/weights/best.pt", f"weights/yolo5_{opt.dataset}_best.pt")
        print(f"Pesos salvos em weights/yolo5_{opt.dataset}_best.pt")

if __name__ == '__main__':
    train()