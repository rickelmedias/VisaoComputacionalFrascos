import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import yaml
import glob
import csv
import numpy as np
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import argparse
import sys
import time
from pathlib import Path

# --- FIX PARA "TOO MANY OPEN FILES" ---
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# --------------------------------------

# --- ARGUMENTOS ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['3k', '7k'])
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--patience', type=int, default=15)
opt = parser.parse_args()

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"--> Rodando em: {DEVICE}")

# --- PATHS ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
YAML_PATH = str(PROJECT_ROOT / "data_3k.yaml")
os.makedirs("metrics", exist_ok=True)
os.makedirs("weights", exist_ok=True)

# --- DATASET ---
class YoloFormatDataset(Dataset):
    def __init__(self, yaml_path, split='train', width=640, height=640, transform=None):
        self.transform = transform
        self.width = width
        self.height = height
        
        if not os.path.exists(yaml_path):
            print(f"ERRO CRÍTICO: {yaml_path} não encontrado!")
            sys.exit(1)
            
        with open(yaml_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        yaml_root = data_cfg.get('path', '.')
        if not os.path.isabs(yaml_root):
            yaml_root = os.path.join(os.getcwd(), yaml_root)
            
        subdir = data_cfg['train'] if split == 'train' else data_cfg['val']
        img_dir = os.path.join(yaml_root, subdir)
        
        if not os.path.exists(img_dir):
            print(f"ERRO: Diretório não encontrado: {img_dir}")
            sys.exit(1)
        
        self.image_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.image_files = [x for x in self.image_files if x.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.webp'))]
        
        if len(self.image_files) == 0:
            sys.exit(1)
            
        self.num_classes = data_cfg['nc'] + 1 

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert("RGB").resize((self.width, self.height))
            img.load()
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))

        label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        cx *= self.width; cy *= self.height
                        w *= self.width; h *= self.height
                        x1 = cx - w/2; y1 = cy - h/2
                        x2 = cx + w/2; y2 = cy + h/2
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(self.width, x2); y2 = min(self.height, y2)
                        if x2 > x1 + 1 and y2 > y1 + 1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls_id + 1)
        
        target = {}
        if boxes:
            target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.tensor(labels, dtype=torch.int64)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            
        target['image_id'] = torch.tensor([idx])
        target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
        target['iscrowd'] = torch.zeros((len(target['labels']),), dtype=torch.int64)

        if self.transform: img = self.transform(img)
        return img, target

    def __len__(self): return len(self.image_files)

# --- UTILS ---
def get_transform(train):
    tr = [T.ToTensor()]
    if train:
        tr += [T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.015), T.RandomHorizontalFlip(0.5)]
    tr.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return T.Compose(tr)

def make_sampler(dataset):
    print("Calculando pesos...")
    count = {}
    img_weights = []
    for idx in range(len(dataset)):
        path = dataset.image_files[idx]
        lpath = path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        classes = []
        if os.path.exists(lpath):
            with open(lpath) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts: classes.append(int(parts[0]) + 1)
        if not classes: classes = [0]
        for c in classes: count[c] = count.get(c, 0) + 1
            
    weight_per_class = {c: 1.0/cnt for c, cnt in count.items() if cnt > 0}
    for idx in range(len(dataset)):
        path = dataset.image_files[idx]
        lpath = path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        classes = []
        if os.path.exists(lpath):
            with open(lpath) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts: classes.append(int(parts[0]) + 1)
        if not classes: w = weight_per_class.get(0, 0.0)
        else: w = max([weight_per_class.get(c, 0) for c in classes])
        img_weights.append(w)
    return WeightedRandomSampler(torch.DoubleTensor(img_weights), len(img_weights))

def collate_fn(batch): return tuple(zip(*batch))

# --- HELPER DE LOSS ---
class Averager:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

# --- MAIN ---
def main():
    print(f"--> Iniciando Faster R-CNN Ultimate ({opt.dataset})...")
    
    train_ds = YoloFormatDataset(YAML_PATH, split='train', transform=get_transform(True))
    val_ds = YoloFormatDataset(YAML_PATH, split='val', transform=get_transform(False))
    
    # Reduz workers se der erro
    sampler = make_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=opt.batch, sampler=sampler, collate_fn=collate_fn, num_workers=opt.workers)
    val_loader = DataLoader(val_ds, batch_size=opt.batch, shuffle=False, collate_fn=collate_fn, num_workers=opt.workers)
    
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, train_ds.num_classes)
    model.to(DEVICE)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    csv_file = f"metrics/faster_ultimate_{opt.dataset}.csv"
    headers = [
        "epoch", "time", 
        "train_loss", "train_box", "train_cls", "train_rpn",
        "val_loss", "val_box", "val_cls", "val_rpn",
        "map50", "map50_95", "recall", "lr"
    ]
    
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(headers)
        
    best_map = 0.0
    patience_cnt = 0
    
    for epoch in range(opt.epochs):
        t0 = time.time()
        
        # --- TREINO ---
        model.train()
        train_loss_avg = Averager()
        train_box_avg = Averager()
        train_cls_avg = Averager()
        train_rpn_avg = Averager()
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch} [Train]")
        for imgs, tgts in pbar:
            imgs = [img.to(DEVICE) for img in imgs]
            tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in tgts]
            
            loss_dict = model(imgs, tgts)
            
            # Decomposição de perdas
            l_cls = loss_dict['loss_classifier']
            l_box = loss_dict['loss_box_reg']
            # RPN = Objectness + RPN Box Reg
            l_rpn = loss_dict['loss_objectness'] + loss_dict['loss_rpn_box_reg']
            
            loss = sum(loss_dict.values())
            
            if not torch.isfinite(loss):
                print(f"AVISO: Loss infinita. Skip.")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Atualiza médias
            train_loss_avg.update(loss.item())
            train_box_avg.update(l_box.item())
            train_cls_avg.update(l_cls.item())
            train_rpn_avg.update(l_rpn.item())
            
            pbar.set_postfix({'loss': f"{loss.item():.2f}"})
            
        # --- VALIDAÇÃO (LOSS) ---
        # Hack: Usar model.train() com torch.no_grad() para extrair loss de validação
        # Faster RCNN não calcula loss em mode eval()
        model.train() 
        val_loss_avg = Averager()
        val_box_avg = Averager()
        val_cls_avg = Averager()
        val_rpn_avg = Averager()
        
        print("Calculando Val Loss...")
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs = [img.to(DEVICE) for img in imgs]
                tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in tgts]
                
                loss_dict = model(imgs, tgts)
                l_cls = loss_dict['loss_classifier']
                l_box = loss_dict['loss_box_reg']
                l_rpn = loss_dict['loss_objectness'] + loss_dict['loss_rpn_box_reg']
                loss = sum(loss_dict.values())
                
                val_loss_avg.update(loss.item())
                val_box_avg.update(l_box.item())
                val_cls_avg.update(l_cls.item())
                val_rpn_avg.update(l_rpn.item())

        # --- VALIDAÇÃO (MÉTRICAS) ---
        model.eval()
        metric = MeanAveragePrecision()
        
        print("Calculando Métricas (mAP)...")
        with torch.no_grad():
            for imgs, tgts in tqdm(val_loader, desc="[Val Metrics]"):
                imgs = [img.to(DEVICE) for img in imgs]
                tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in tgts]
                preds = model(imgs)
                metric.update(preds, tgts)
        
        try:
            m = metric.compute()
            map50 = m['map_50'].item()
            map50_95 = m['map'].item() # Equivalente ao mAP50-95 do YOLO
            recall = m['mar_100'].item()
        except:
            map50 = 0.0; map50_95 = 0.0; recall = 0.0
            
        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print formatado estilo YOLO
        print(f"\nResultados Ep {epoch} ({epoch_time:.1f}s):")
        print(f"Train Loss: {train_loss_avg.avg:.4f} (Box: {train_box_avg.avg:.3f} | Cls: {train_cls_avg.avg:.3f})")
        print(f"Val Loss:   {val_loss_avg.avg:.4f} (Box: {val_box_avg.avg:.3f} | Cls: {val_cls_avg.avg:.3f})")
        print(f"Métricas:   mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f} | Recall: {recall:.4f}")
        
        # Log CSV Rico
        row = [
            epoch, f"{epoch_time:.2f}",
            f"{train_loss_avg.avg:.5f}", f"{train_box_avg.avg:.5f}", f"{train_cls_avg.avg:.5f}", f"{train_rpn_avg.avg:.5f}",
            f"{val_loss_avg.avg:.5f}", f"{val_box_avg.avg:.5f}", f"{val_cls_avg.avg:.5f}", f"{val_rpn_avg.avg:.5f}",
            f"{map50:.5f}", f"{map50_95:.5f}", f"{recall:.5f}", f"{current_lr:.6f}"
        ]
        
        with open(csv_file, 'a', newline='') as f:
            csv.writer(f).writerow(row)
            
        lr_scheduler.step(map50_95)
        
        if map50_95 > best_map:
            best_map = map50_95
            patience_cnt = 0
            torch.save(model.state_dict(), f"weights/faster_{opt.dataset}_best.pth")
            print("--> NEW BEST MODEL SAVED!")
        else:
            patience_cnt += 1
            
        if patience_cnt >= opt.patience:
            print("Early Stopping trigger!")
            break
            
        torch.save(model.state_dict(), f"weights/faster_{opt.dataset}_last.pth")

if __name__ == "__main__":
    main()