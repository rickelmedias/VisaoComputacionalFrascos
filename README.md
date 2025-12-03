# 🔬 Visão Computacional - Detecção de Frascos

Este projeto implementa e compara diferentes arquiteturas de detecção de objetos (YOLOv5, YOLO11 e Faster R-CNN) para classificação de frascos em 4 categorias: **Danificado**, **Frasco**, **Lacrado** e **Sem_lacre**.

## 📁 Estrutura do Projeto

```
VisaoComputacional-Frascos/
├── abordagem_1/           # Primeira abordagem de treinamento
│   ├── faster/            # Faster R-CNN
│   ├── yolo5/             # YOLOv5
│   └── yolo11/            # YOLO11
├── abordagem_2/           # Segunda abordagem (usa config.yaml)
│   ├── faster/
│   ├── yolo5/
│   ├── yolo11/
│   └── config.yaml        # Configurações para YOLO (abordagem 2)
├── datasets/              # ⚠️ NÃO INCLUÍDO (baixar separadamente)
│   ├── dataset3k/         # Dataset com ~3k imagens de treino
│   └── dataset7k/         # Dataset com ~7k imagens de treino
├── data_3k.yaml           # Configuração do dataset 3k
├── data_7k.yaml           # Configuração do dataset 7k
├── .gitignore
├── .gitattributes         # Configuração Git LFS
└── README.md
```

## 🚀 Setup Inicial

### 1. Clone o Repositório

```bash
git clone https://github.com/SEU_USUARIO/VisaoComputacional-Frascos.git
cd VisaoComputacional-Frascos
```

### 2. Instale o Git LFS (para baixar os modelos treinados)

Os arquivos `.pt` e `.pth` (modelos treinados) são armazenados via Git LFS devido ao seu tamanho.

```bash
# Ubuntu/Debian
sudo apt install git-lfs

# macOS
brew install git-lfs

# Após instalar, inicialize no repositório
git lfs install
git lfs pull
```

### 3. Instale as Dependências

```bash
pip install torch torchvision ultralytics pyyaml tqdm torchmetrics pandas pillow numpy
```

### 4. Baixe os Datasets

Os datasets não estão incluídos no repositório. Baixe-os e coloque na pasta `datasets/`:

```
datasets/
├── dataset3k/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
└── dataset7k/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
```

## 🏋️ Treinamento

### Abordagem 1

Os scripts da abordagem 1 usam argumentos de linha de comando:

```bash
# YOLOv5 com dataset 3k
cd abordagem_1/yolo5
python yolo5_abordagem_1_3k.py --dataset 3k --epochs 100 --batch 16

# YOLOv5 com dataset 7k
python yolo5_abordagem_1_7k.py --dataset 7k --epochs 100 --batch 16

# YOLO11 com dataset 3k
cd ../yolo11
python yolo11_abordagem_1_3k.py --dataset 3k --epochs 100 --batch 16

# Faster R-CNN com dataset 3k
cd ../faster
python faster_abordagem_1_3k.py --dataset 3k --epochs 100 --batch 16
```

### Abordagem 2

Os scripts da abordagem 2 usam o `config.yaml` centralizado:

```bash
# YOLOv5 com dataset 3k
cd abordagem_2/yolo5
python yolo5_abordagem_2_3k.py

# YOLO11 com dataset 7k
cd ../yolo11
python yolo11_abordagem_2_7k.py

# Faster R-CNN com dataset 3k
cd ../faster
python faster_abordagem_2_3k.py
```

## 📊 Arquivos de Resultados

Cada modelo gera:

- **`*_best.pt` / `*_best.pth`**: Melhor modelo durante o treinamento
- **`*_results.csv`**: Métricas por época (loss, mAP, recall, etc.)

## 🗃️ Classes do Dataset

| ID | Classe     | Descrição                    |
|----|------------|------------------------------|
| 0  | Danificado | Frasco com danos visíveis    |
| 1  | Frasco     | Frasco genérico              |
| 2  | Lacrado    | Frasco com lacre intacto     |
| 3  | Sem_lacre  | Frasco sem lacre             |

## ⚠️ Sobre Arquivos Grandes

Este repositório usa **Git LFS** para gerenciar arquivos grandes (`.pt`, `.pth`).

Se você clonou o repositório e os arquivos de modelo aparecem pequenos (~130 bytes), execute:

```bash
git lfs pull
```

### Alternativa: Download Manual

Se Git LFS não funcionar, os modelos também podem ser disponibilizados via:
- GitHub Releases
- Google Drive
- Hugging Face Hub

## 📝 Configurações

### abordagem_2/config.yaml

```yaml
epochs: 500
batch: 16
patience: 30
imgsz: 640
# ... (veja o arquivo completo)
```

### data_3k.yaml / data_7k.yaml

```yaml
path: datasets/dataset3k  # ou dataset7k
train: train/images
val: val/images
nc: 4
names: ['Danificado', 'Frasco', 'Lacrado', 'Sem_lacre']
```

## 🛠️ Requisitos

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (para GPU)
- Git LFS

## 📜 Licença

Este projeto é parte de um trabalho acadêmico de Visão Computacional.

