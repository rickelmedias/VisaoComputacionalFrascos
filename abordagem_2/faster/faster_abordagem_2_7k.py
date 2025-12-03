import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import os
import glob
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pathlib import Path
import gc

# --- PATHS ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_YAML = str(PROJECT_ROOT / "data_7k.yaml")
OUTPUT_DIR = PROJECT_ROOT / "abordagem_2" / "faster" / "weights_7k"

# --- CONFIGURAÇÕES OTIMIZADAS PARA A100 ---
BATCH_SIZE = 24
NUM_WORKERS = 16
PREFETCH_FACTOR = 2
NUM_EPOCHS = 500
PATIENCE = 30
LEARNING_RATE = 0.01
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Otimizações para A100
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. DATASET ---
class YoloFormatDataset(Dataset):
    def __init__(self, yaml_path, split='train', transforms=None):
        self.transforms = transforms
        with open(yaml_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        base_path = self.cfg.get('path', '.')
        if not os.path.isabs(base_path):
            base_path = os.path.join(PROJECT_ROOT, base_path)
            
        img_dir = os.path.join(base_path, self.cfg.get(split))
        self.images = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.images = [x for x in self.images if x.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.classes = self.cfg['names']

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    
                    x1 = (cx - bw/2) * w
                    y1 = (cy - bh/2) * h
                    x2 = (cx + bw/2) * w
                    y2 = (cy + bh/2) * h
                    
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w, x2); y2 = min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls_id + 1)

        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)

# --- 2. TRANSFORMS ---
def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

# --- 3. MODELO ---
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- 4. TREINO COM GESTÃO DE MEMÓRIA ---
def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler):
    model.train()
    losses_sum = {'loss_classifier': 0, 'loss_box_reg': 0, 'loss_objectness': 0, 'loss_rpn_box_reg': 0}
    total_loss_sum = 0
    steps = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, (images, targets) in enumerate(pbar):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss_sum += losses.item()
        for k, v in loss_dict.items():
            if k in losses_sum:
                losses_sum[k] += v.item()
        steps += 1
        
        pbar.set_postfix({'loss': f"{losses.item():.4f}"})
        
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    avg_losses = {k: v / max(1, steps) for k, v in losses_sum.items()}
    avg_total = total_loss_sum / max(1, steps)
    
    yolo_box = avg_losses['loss_box_reg'] + avg_losses['loss_rpn_box_reg']
    yolo_cls = avg_losses['loss_classifier'] + avg_losses['loss_objectness']
    
    return avg_total, yolo_box, yolo_cls

# --- 5. VALIDAÇÃO COM GESTÃO DE MEMÓRIA ---
@torch.no_grad()
def evaluate_full(model, data_loader, device):
    torch.cuda.empty_cache()
    gc.collect()
    
    model.train()
    val_loss_sum = 0
    val_box_sum = 0
    val_cls_sum = 0
    steps = 0
    
    for images, targets in data_loader:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        
        with torch.amp.autocast('cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        val_loss_sum += losses.item()
        val_box_sum += (loss_dict['loss_box_reg'] + loss_dict['loss_rpn_box_reg']).item()
        val_cls_sum += (loss_dict['loss_classifier'] + loss_dict['loss_objectness']).item()
        steps += 1
    
    avg_val_loss = val_loss_sum / max(1, steps)
    avg_val_box = val_box_sum / max(1, steps)
    avg_val_cls = val_cls_sum / max(1, steps)

    model.eval()
    metric = MeanAveragePrecision(class_metrics=False)
    
    for images, targets in tqdm(data_loader, desc="[Val Metrics]", leave=False):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            
        preds = []
        for out in outputs:
            preds.append({
                'boxes': out['boxes'],
                'scores': out['scores'],
                'labels': out['labels']
            })
        metric.update(preds, targets)
        
    m_dict = metric.compute()
    map50 = m_dict['map_50'].item()
    map50_95 = m_dict['map'].item()
    recall = m_dict['mar_100'].item()
    precision = 0.0 

    return avg_val_loss, avg_val_box, avg_val_cls, precision, recall, map50, map50_95

# --- MAIN ---
def main():
    print(f"=== Faster R-CNN OTIMIZADO - Abordagem 2 - Dataset 7k ===")
    print(f"Config: {NUM_EPOCHS} Epochs | Batch {BATCH_SIZE} | Workers {NUM_WORKERS}")
    print(f"Device: {DEVICE} | CUDA: {torch.cuda.is_available()}")
    print(f"Dataset: {DATA_YAML}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    dataset = YoloFormatDataset(DATA_YAML, split='train', transforms=get_transform(True))
    dataset_val = YoloFormatDataset(DATA_YAML, split='val', transforms=get_transform(False))
    
    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR
    )
    val_loader = DataLoader(
        dataset_val, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR
    )

    num_classes = dataset.cfg['nc'] + 1
    model = get_model(num_classes).to(DEVICE)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    
    # Scheduler híbrido: CosineAnnealingWarmRestarts permite aprendizado contínuo
    # com reinícios periódicos do LR, evitando estagnação prematura
    # T_0=20: reinicia a cada 20 épocas inicialmente
    # T_mult=2: dobra o período a cada reinício (20, 40, 80...)
    # eta_min: LR mínimo para não parar de aprender completamente
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    scaler = torch.amp.GradScaler('cuda')

    csv_path = OUTPUT_DIR / "results.csv"
    cols = [
        "epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss", 
        "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        "val/box_loss", "val/cls_loss", "val/dfl_loss", 
        "lr/pg0", "lr/pg1", "lr/pg2"
    ]
    if not csv_path.exists():
        pd.DataFrame(columns=cols).to_csv(csv_path, index=False)

    best_map = 0.0
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        t_loss, t_box, t_cls = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, scaler)
        v_loss, v_box, v_cls, prec, rec, map50, map95 = evaluate_full(model, val_loader, DEVICE)
        
        lr_scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']

        print(f"Ep {epoch} | mAP@50-95: {map95:.4f} (Best: {best_map:.4f}) | Patience: {epochs_no_improve}/{PATIENCE}")
        
        row = [
            epoch, t_box, t_cls, 0.0,
            prec, rec, map50, map95,
            v_box, v_cls, 0.0,
            curr_lr, curr_lr, curr_lr
        ]
        pd.DataFrame([row], columns=cols).to_csv(csv_path, mode='a', header=False, index=False)

        if map95 > best_map:
            best_map = map95
            epochs_no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best.pth")
            print("--> Novo Recorde! Modelo salvo.")
        else:
            epochs_no_improve += 1
        
        torch.save(model.state_dict(), OUTPUT_DIR / "last.pth")

        if epochs_no_improve >= PATIENCE:
            print(f"\n--> Early Stopping acionado na época {epoch}!")
            break
        
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
