#!/usr/bin/env python3
"""
==============================================================================
🔬 AVALIAÇÃO COMPLETA DOS MODELOS - ABORDAGEM 2
==============================================================================
Script para avaliar os modelos treinados (YOLOv5, YOLO11 e Faster R-CNN)
gerando métricas, gráficos, matriz de confusão e matriz de correlação.

Uso:
    python avaliar_modelos.py

Autor: Gerado para análise de Visão Computacional - Detecção de Frascos
==============================================================================
"""

import os
import sys
import glob
import yaml
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ==============================================================================
# 📌 CONFIGURAÇÕES
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
ABORDAGEM_DIR = Path(__file__).parent
OUTPUT_DIR = ABORDAGEM_DIR / "avaliacoes"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configurações visuais
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'figure.titlesize': 14,
    'figure.figsize': (12, 8),
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"🖥️  Dispositivo: {DEVICE}")

# Classes do dataset
CLASS_NAMES = ['Danificado', 'Frasco', 'Lacrado', 'Sem_lacre']
N_CLASSES = len(CLASS_NAMES)

# ==============================================================================
# 🧮 FUNÇÕES AUXILIARES
# ==============================================================================

def box_iou(box1, box2):
    """Calcula IoU (Interseção sobre União) entre caixas"""
    if len(box1) == 0 or len(box2) == 0:
        return np.array([])
    
    lt = np.maximum(box1[:, None, :2], box2[:, :2])
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def load_faster_rcnn(weights_path, num_classes):
    """Carrega o Faster R-CNN com a arquitetura correta"""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


def load_yolo_model(weights_path):
    """Carrega modelo YOLO (v5 ou v11)"""
    from ultralytics import YOLO
    model = YOLO(weights_path)
    return model


def get_predictions(model, model_type, img_path, conf_threshold=0.5):
    """Faz inferência unificada (YOLO ou Faster)"""
    img = Image.open(img_path).convert("RGB")
    
    if model_type == 'yolo':
        results = model(img, verbose=False, conf=conf_threshold)
        if not results or len(results[0].boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        scores = results[0].boxes.conf.cpu().numpy()
    
    elif model_type == 'faster':
        img_t = T.ToTensor()(img).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            out = model(img_t)[0]
        keep = out['scores'] > conf_threshold
        boxes = out['boxes'][keep].cpu().numpy()
        classes = out['labels'][keep].cpu().numpy() - 1  # Faster é 1-based
        scores = out['scores'][keep].cpu().numpy()
    
    return boxes, classes, scores


def load_ground_truth(label_path, img_w, img_h):
    """Carrega anotações ground truth de um arquivo YOLO format"""
    gt_boxes, gt_classes = [], []
    
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    x1 = (cx - w/2) * img_w
                    y1 = (cy - h/2) * img_h
                    x2 = (cx + w/2) * img_w
                    y2 = (cy + h/2) * img_h
                    gt_boxes.append([x1, y1, x2, y2])
                    gt_classes.append(c)
    
    return np.array(gt_boxes), np.array(gt_classes)


def classify_overfitting(train_loss, val_loss):
    """Classifica o nível de overfitting baseado na razão val/train loss"""
    if train_loss is None or val_loss is None or train_loss == 0:
        return "N/A"
    
    ratio = val_loss / train_loss
    if ratio <= 1.2:
        return "✅ Baixo"
    elif ratio <= 1.5:
        return "⚠️ Médio"
    else:
        return "❌ Alto"


def classify_stability(std_value):
    """Classifica a estabilidade baseado no desvio padrão do mAP"""
    if std_value is None:
        return "N/A"
    
    if std_value <= 0.01:
        return "⭐ Alta"
    elif std_value <= 0.03:
        return "🔶 Média"
    else:
        return "⚠️ Baixa"


# ==============================================================================
# 📊 ANÁLISE PROFUNDA DO HISTÓRICO DE TREINAMENTO (CSV)
# ==============================================================================

def analyze_training_history_deep(csv_path, model_name, output_dir):
    """Analisa CSV de histórico de treinamento com métricas profundas"""
    print(f"\n📈 Analisando histórico: {model_name}")
    
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"   ❌ Erro ao ler CSV: {e}")
        return None
    
    # Identifica colunas relevantes
    epoch_col = 'epoch' if 'epoch' in df.columns else df.columns[0]
    
    # Encontra colunas de métricas
    map_col = next((c for c in df.columns if 'map50-95' in c.lower() or 'map50_95' in c.lower()), None)
    map50_col = next((c for c in df.columns if ('map50' in c.lower() and '95' not in c) or 'map50(b)' in c.lower()), None)
    train_loss_cols = [c for c in df.columns if 'train' in c.lower() and 'loss' in c.lower()]
    val_loss_cols = [c for c in df.columns if 'val' in c.lower() and 'loss' in c.lower()]
    prec_col = next((c for c in df.columns if 'precision' in c.lower()), None)
    rec_col = next((c for c in df.columns if 'recall' in c.lower()), None)
    lr_col = next((c for c in df.columns if 'lr' in c.lower()), None)
    
    # === GRÁFICOS DE HISTÓRICO ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Histórico de Treinamento: {model_name}', fontsize=16, fontweight='bold')
    
    # Gráfico 1: mAP
    ax1 = axes[0, 0]
    if map_col:
        ax1.plot(df[epoch_col], df[map_col], 'b-', linewidth=2, label='mAP@50-95')
    if map50_col:
        ax1.plot(df[epoch_col], df[map50_col], 'g--', linewidth=2, label='mAP@50')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('mAP')
    ax1.set_title('Evolução do mAP')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Loss de Treino
    ax2 = axes[0, 1]
    for col in train_loss_cols[:3]:
        label = col.replace('train/', '').replace('_', ' ')
        ax2.plot(df[epoch_col], df[col], linewidth=1.5, label=label)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss de Treinamento')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Loss de Validação
    ax3 = axes[1, 0]
    for col in val_loss_cols[:3]:
        label = col.replace('val/', '').replace('_', ' ')
        ax3.plot(df[epoch_col], df[col], linewidth=1.5, label=label)
    ax3.set_xlabel('Época')
    ax3.set_ylabel('Loss')
    ax3.set_title('Loss de Validação')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Precision e Recall
    ax4 = axes[1, 1]
    if prec_col:
        ax4.plot(df[epoch_col], df[prec_col], 'b-', linewidth=2, label='Precision')
    if rec_col:
        ax4.plot(df[epoch_col], df[rec_col], 'r-', linewidth=2, label='Recall')
    ax4.set_xlabel('Época')
    ax4.set_ylabel('Valor')
    ax4.set_title('Precision e Recall')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / f"historico_{model_name}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"   ✅ Gráfico de histórico salvo: {save_path}")
    
    # === CÁLCULO DE MÉTRICAS ===
    best_idx = df[map_col].idxmax() if map_col else 0
    best_row = df.loc[best_idx]
    
    # Loss final e da melhor época
    final_train_loss = df[train_loss_cols[0]].iloc[-1] if train_loss_cols else None
    final_val_loss = df[val_loss_cols[0]].iloc[-1] if val_loss_cols else None
    best_train_loss = best_row[train_loss_cols[0]] if train_loss_cols else None
    best_val_loss = best_row[val_loss_cols[0]] if val_loss_cols else None
    
    # Estabilidade (desvio padrão das últimas 10 épocas)
    stability_std = df[map_col].tail(10).std() if map_col else None
    
    # Overfitting na melhor época
    overfitting_value = (best_val_loss - best_train_loss) if best_train_loss and best_val_loss else None
    
    stats = {
        'modelo': model_name,
        'epochs': len(df),
        'best_epoch': int(best_row[epoch_col]),
        'best_map50_95': round(best_row[map_col], 4) if map_col else None,
        'best_map50': round(best_row[map50_col], 4) if map50_col else None,
        'precision': round(best_row[prec_col], 4) if prec_col else None,
        'recall': round(best_row[rec_col], 4) if rec_col else None,
        'best_train_loss': round(best_train_loss, 6) if best_train_loss else None,
        'best_val_loss': round(best_val_loss, 6) if best_val_loss else None,
        'overfitting_value': round(overfitting_value, 6) if overfitting_value else None,
        'overfitting_class': classify_overfitting(best_train_loss, best_val_loss),
        'stability_std': round(stability_std, 6) if stability_std else None,
        'stability_class': classify_stability(stability_std),
    }
    
    return stats, df




# ==============================================================================
# 📊 ANÁLISE COMPARATIVA DE EVOLUÇÃO
# ==============================================================================

def generate_evolution_comparison(all_dfs, output_dir):
    """Gera gráfico comparativo da evolução de todos os modelos"""
    print("\n📊 Gerando Comparativo de Evolução...")
    
    if not all_dfs:
        return
    
    # Separa por dataset (3k e 7k)
    for dataset_size in ['3k', '7k']:
        subset_dfs = {k: v for k, v in all_dfs.items() if dataset_size in k}
        
        if not subset_dfs:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Comparativo de Evolução - Dataset {dataset_size.upper()}', fontsize=14, fontweight='bold')
        
        # Gráfico 1: Evolução mAP@50-95
        ax1 = axes[0]
        for model_name, df in subset_dfs.items():
            map_col = next((c for c in df.columns if 'map50-95' in c.lower() or 'map50_95' in c.lower()), None)
            if map_col:
                ax1.plot(df['epoch'], df[map_col], linewidth=2, label=model_name)
        
        ax1.set_xlabel('Época')
        ax1.set_ylabel('mAP@50-95')
        ax1.set_title('Evolução do mAP@50-95')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Evolução da Loss de Validação
        ax2 = axes[1]
        for model_name, df in subset_dfs.items():
            val_loss_col = next((c for c in df.columns if 'val' in c.lower() and 'box' in c.lower()), None)
            if val_loss_col:
                ax2.plot(df['epoch'], df[val_loss_col], linewidth=2, label=model_name)
        
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Validation Box Loss')
        ax2.set_title('Evolução da Loss de Validação')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = output_dir / f"evolucao_comparativa_{dataset_size}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"   ✅ Comparativo {dataset_size} salvo: {save_path}")


# ==============================================================================
# 📋 TABELA RESUMO FINAL
# ==============================================================================

def generate_summary_table(all_stats, output_dir):
    """Gera tabela resumo com todas as métricas solicitadas"""
    print("\n📋 Gerando Tabela Resumo Final...")
    
    if not all_stats:
        return None
    
    valid_stats = [s for s in all_stats if s]
    
    # Cria DataFrame com formato específico
    summary_data = []
    for stat in valid_stats:
        summary_data.append({
            'Modelo': stat['modelo'],
            'Melhor Época': stat.get('best_epoch', 'N/A'),
            'mAP@50': f"{stat.get('best_map50', 0)*100:.2f}%" if stat.get('best_map50') else 'N/A',
            'mAP@50-95': f"{stat.get('best_map50_95', 0)*100:.2f}%" if stat.get('best_map50_95') else 'N/A',
            'Overfitting': stat.get('overfitting_class', 'N/A'),
            'Estabilidade': stat.get('stability_class', 'N/A'),
            'Precision': f"{stat.get('precision', 0)*100:.2f}%" if stat.get('precision') else 'N/A',
            'Recall': f"{stat.get('recall', 0)*100:.2f}%" if stat.get('recall') else 'N/A',
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Ordena por mAP@50-95 (convertendo de volta para número para ordenação)
    df_summary['_sort'] = df_summary['mAP@50-95'].apply(
        lambda x: float(x.replace('%', '')) if x != 'N/A' else 0
    )
    df_summary = df_summary.sort_values('_sort', ascending=False).drop('_sort', axis=1)
    
    # Salva CSV
    csv_path = output_dir / "tabela_resumo_modelos.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"   ✅ Tabela resumo salva: {csv_path}")
    
    # Exibe no console
    print("\n" + "="*100)
    print("📊 TABELA RESUMO - TODOS OS MODELOS")
    print("="*100)
    print(df_summary.to_string(index=False))
    print("="*100)
    
    return df_summary


# ==============================================================================
# 📊 GRÁFICO DE BARRAS COMPARATIVO
# ==============================================================================

def generate_bar_comparison(all_stats, output_dir):
    """Gera gráfico de barras comparativo final"""
    print("\n📊 Gerando Gráfico Comparativo Final...")
    
    if not all_stats:
        return
    
    valid_stats = [s for s in all_stats if s and s.get('best_map50_95')]
    
    if not valid_stats:
        return
    
    # Prepara dados
    models = [s['modelo'] for s in valid_stats]
    map50_95 = [s.get('best_map50_95', 0) for s in valid_stats]
    map50 = [s.get('best_map50', 0) for s in valid_stats]
    precision = [s.get('precision', 0) or 0 for s in valid_stats]
    recall = [s.get('recall', 0) or 0 for s in valid_stats]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - 1.5*width, map50_95, width, label='mAP@50-95', color='#3498db')
    bars2 = ax.bar(x - 0.5*width, map50, width, label='mAP@50', color='#2ecc71')
    bars3 = ax.bar(x + 0.5*width, precision, width, label='Precision', color='#e74c3c')
    bars4 = ax.bar(x + 1.5*width, recall, width, label='Recall', color='#f39c12')
    
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparativo Final de Métricas - Abordagem 2', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adiciona valores nas barras
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    save_path = output_dir / "comparativo_barras_final.png"
    plt.savefig(save_path)
    plt.close()
    print(f"   ✅ Gráfico de barras salvo: {save_path}")


# ==============================================================================
# 🎯 MATRIZ DE CONFUSÃO (quando dataset disponível)
# ==============================================================================

def evaluate_model_with_confusion(weights_path, model_type, dataset_yaml, model_name, output_dir, iou_threshold=0.5, conf_threshold=0.5):
    """Avalia modelo gerando matriz de confusão (requer dataset)"""
    print(f"\n🎯 Avaliando modelo: {model_name}")
    
    # Carrega configuração do dataset
    with open(dataset_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Tenta encontrar o diretório de validação
    val_dirs = [
        os.path.join(cfg['path'], cfg['val']),
        os.path.join(PROJECT_ROOT, cfg['path'], cfg['val']),
    ]
    
    val_dir = None
    for vd in val_dirs:
        if os.path.exists(vd):
            val_dir = vd
            break
    
    if val_dir is None:
        print(f"   ⚠️ Dataset não encontrado. Matriz de confusão não será gerada.")
        print(f"   ℹ️ Esperado em: {cfg['path']}")
        return None
    
    img_files = sorted(glob.glob(os.path.join(val_dir, "*.*")))
    img_files = [x for x in img_files if x.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not img_files:
        print(f"   ❌ Nenhuma imagem encontrada em {val_dir}")
        return None
    
    print(f"   📁 Encontradas {len(img_files)} imagens de validação")
    
    # Carrega modelo
    try:
        if model_type == 'yolo':
            model = load_yolo_model(weights_path)
        else:
            model = load_faster_rcnn(weights_path, N_CLASSES)
    except Exception as e:
        print(f"   ❌ Erro ao carregar modelo: {e}")
        return None
    
    # Matriz de confusão
    matrix = np.zeros((N_CLASSES + 1, N_CLASSES + 1), dtype=int)
    
    for img_file in tqdm(img_files, desc=f"   Validando {model_name}", leave=False):
        img = Image.open(img_file)
        img_w, img_h = img.size
        label_file = img_file.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        gt_boxes, gt_classes = load_ground_truth(label_file, img_w, img_h)
        
        pred_boxes, pred_classes, pred_scores = get_predictions(model, model_type, img_file, conf_threshold)
        
        matched_gt = set()
        matched_pred = set()
        
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            ious = box_iou(gt_boxes, pred_boxes)
            
            for g_idx, g_cls in enumerate(gt_classes):
                best_iou = 0
                best_p_idx = -1
                
                for p_idx in range(len(pred_classes)):
                    if p_idx in matched_pred:
                        continue
                    if ious[g_idx, p_idx] > best_iou:
                        best_iou = ious[g_idx, p_idx]
                        best_p_idx = p_idx
                
                if best_iou >= iou_threshold:
                    p_cls = pred_classes[best_p_idx]
                    matrix[g_cls, p_cls] += 1
                    matched_gt.add(g_idx)
                    matched_pred.add(best_p_idx)
        
        for g_idx, g_cls in enumerate(gt_classes):
            if g_idx not in matched_gt:
                matrix[g_cls, N_CLASSES] += 1
        
        for p_idx, p_cls in enumerate(pred_classes):
            if p_idx not in matched_pred:
                matrix[N_CLASSES, p_cls] += 1
    
    # Plota matriz de confusão
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = CLASS_NAMES + ['Background']
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'Matriz de Confusão: {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Real (Ground Truth)')
    ax.set_xlabel('Predito')
    plt.tight_layout()
    
    save_path = output_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"   ✅ Matriz de confusão salva: {save_path}")
    
    return matrix


# ==============================================================================
# 🚀 FUNÇÃO PRINCIPAL
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("🔬 AVALIAÇÃO COMPLETA - ABORDAGEM 2")
    print("="*60)
    print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Diretório de saída: {OUTPUT_DIR}")
    
    all_stats = []
    all_dfs = {}
    
    # Define os modelos a avaliar
    models_config = [
        # YOLOv5
        {'csv': ABORDAGEM_DIR / 'yolo5' / 'yolo5_results_abordagem_2_3k.csv', 'weights': ABORDAGEM_DIR / 'yolo5' / 'yolo5_best_abordagem_2_3k.pt', 'type': 'yolo', 'dataset': '3k', 'name': 'YOLOv5_3k'},
        {'csv': ABORDAGEM_DIR / 'yolo5' / 'yolo5_results_abordagem_2_7k.csv', 'weights': ABORDAGEM_DIR / 'yolo5' / 'yolo5_best_abordagem_2_7k.pt', 'type': 'yolo', 'dataset': '7k', 'name': 'YOLOv5_7k'},
        # YOLO11
        {'csv': ABORDAGEM_DIR / 'yolo11' / 'yolo11_results_abordagem_2_3k.csv', 'weights': ABORDAGEM_DIR / 'yolo11' / 'yolo11_best_abordagem_2_3k.pt', 'type': 'yolo', 'dataset': '3k', 'name': 'YOLO11_3k'},
        {'csv': ABORDAGEM_DIR / 'yolo11' / 'yolo11_results_abordagem_2_7k.csv', 'weights': ABORDAGEM_DIR / 'yolo11' / 'yolo11_best_abordagem_2_7k.pt', 'type': 'yolo', 'dataset': '7k', 'name': 'YOLO11_7k'},
        # Faster R-CNN
        {'csv': ABORDAGEM_DIR / 'faster' / 'faster_results_abordagem_2_3k.csv', 'weights': ABORDAGEM_DIR / 'faster' / 'faster_best_abordagem_2_3k.pth', 'type': 'faster', 'dataset': '3k', 'name': 'FasterRCNN_3k'},
        {'csv': ABORDAGEM_DIR / 'faster' / 'faster_results_abordagem_2_7k.csv', 'weights': ABORDAGEM_DIR / 'faster' / 'faster_best_abordagem_2_7k.pth', 'type': 'faster', 'dataset': '7k', 'name': 'FasterRCNN_7k'},
    ]
    
    # 1. Analisa histórico de cada modelo
    for config in models_config:
        csv_path = config['csv']
        model_name = config['name']
        
        if csv_path.exists():
            stats, df = analyze_training_history_deep(csv_path, model_name, OUTPUT_DIR)
            if stats:
                all_stats.append(stats)
            if df is not None:
                all_dfs[model_name] = df
        else:
            print(f"\n⚠️ CSV não encontrado: {csv_path}")
    
    # 2. Gera análises comparativas
    generate_evolution_comparison(all_dfs, OUTPUT_DIR)
    generate_bar_comparison(all_stats, OUTPUT_DIR)
    
    # 3. Gera tabela resumo final (CSV solicitado)
    df_summary = generate_summary_table(all_stats, OUTPUT_DIR)
    
    # 4. Tenta gerar matrizes de confusão (se dataset disponível)
    print("\n" + "="*60)
    print("🎯 TENTANDO GERAR MATRIZES DE CONFUSÃO (requer dataset)")
    print("="*60)
    
    for config in models_config:
        weights_path = config['weights']
        model_type = config['type']
        dataset = config['dataset']
        model_name = config['name']
        yaml_path = PROJECT_ROOT / f"data_{dataset}.yaml"
        
        if weights_path.exists():
            evaluate_model_with_confusion(
                weights_path=str(weights_path),
                model_type=model_type,
                dataset_yaml=str(yaml_path),
                model_name=model_name,
                output_dir=OUTPUT_DIR
            )
    
    # 5. Salva histórico detalhado
    if all_stats:
        df_detailed = pd.DataFrame(all_stats)
        csv_path = OUTPUT_DIR / "historico_detalhado.csv"
        df_detailed.to_csv(csv_path, index=False)
        print(f"\n✅ Histórico detalhado salvo: {csv_path}")
    
    print("\n" + "="*60)
    print("✅ AVALIAÇÃO CONCLUÍDA!")
    print(f"📁 Resultados salvos em: {OUTPUT_DIR}")
    print("="*60)
    
    return all_stats, all_dfs


if __name__ == "__main__":
    stats, dfs = main()
