#!/usr/bin/env python3
"""
==============================================================================
🔬 AVALIAÇÃO COMPLETA DOS MODELOS - ABORDAGEM 2
==============================================================================
Script para avaliar os modelos treinados (YOLOv5, YOLO11 e Faster R-CNN)
gerando métricas, gráficos e matriz de confusão.

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
from sklearn.metrics import classification_report, confusion_matrix

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


# ==============================================================================
# 📊 ANÁLISE DE HISTÓRICO DE TREINAMENTO (CSV)
# ==============================================================================

def analyze_training_history(csv_path, model_name, output_dir):
    """Analisa CSV de histórico de treinamento e gera gráficos"""
    print(f"\n📈 Analisando histórico: {model_name}")
    
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"   ❌ Erro ao ler CSV: {e}")
        return None
    
    # Identifica colunas relevantes
    epoch_col = 'epoch' if 'epoch' in df.columns else df.columns[0]
    
    # Detecta tipo de modelo pelo formato das colunas
    is_yolo = any('metrics' in c.lower() for c in df.columns)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Histórico de Treinamento: {model_name}', fontsize=16, fontweight='bold')
    
    # Gráfico 1: mAP
    ax1 = axes[0, 0]
    map_cols = [c for c in df.columns if 'map50-95' in c.lower() or 'map50_95' in c.lower() or c == 'map50_95']
    map50_cols = [c for c in df.columns if 'map50(b)' in c.lower() or c == 'map50']
    
    if map_cols:
        ax1.plot(df[epoch_col], df[map_cols[0]], 'b-', linewidth=2, label='mAP@50-95')
    if map50_cols:
        ax1.plot(df[epoch_col], df[map50_cols[0]], 'g--', linewidth=2, label='mAP@50')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('mAP')
    ax1.set_title('Evolução do mAP')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Loss de Treino
    ax2 = axes[0, 1]
    train_loss_cols = [c for c in df.columns if 'train' in c.lower() and 'loss' in c.lower()]
    for col in train_loss_cols[:3]:  # Máximo 3 losses
        label = col.replace('train/', '').replace('_', ' ')
        ax2.plot(df[epoch_col], df[col], linewidth=1.5, label=label)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss de Treinamento')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Loss de Validação
    ax3 = axes[1, 0]
    val_loss_cols = [c for c in df.columns if 'val' in c.lower() and 'loss' in c.lower()]
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
    prec_cols = [c for c in df.columns if 'precision' in c.lower()]
    rec_cols = [c for c in df.columns if 'recall' in c.lower()]
    
    if prec_cols:
        ax4.plot(df[epoch_col], df[prec_cols[0]], 'b-', linewidth=2, label='Precision')
    if rec_cols:
        ax4.plot(df[epoch_col], df[rec_cols[0]], 'r-', linewidth=2, label='Recall')
    ax4.set_xlabel('Época')
    ax4.set_ylabel('Valor')
    ax4.set_title('Precision e Recall')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salva gráfico
    save_path = output_dir / f"historico_{model_name}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"   ✅ Gráfico salvo: {save_path}")
    
    # Retorna estatísticas
    stats = {
        'modelo': model_name,
        'epochs': len(df),
        'best_map50_95': df[map_cols[0]].max() if map_cols else None,
        'best_map50': df[map50_cols[0]].max() if map50_cols else None,
        'final_train_loss': df[train_loss_cols[0]].iloc[-1] if train_loss_cols else None,
        'final_val_loss': df[val_loss_cols[0]].iloc[-1] if val_loss_cols else None,
    }
    
    return stats


# ==============================================================================
# 🎯 MATRIZ DE CONFUSÃO E MÉTRICAS
# ==============================================================================

def evaluate_model(weights_path, model_type, dataset_yaml, model_name, output_dir, iou_threshold=0.5, conf_threshold=0.5):
    """Avalia modelo gerando matriz de confusão e métricas detalhadas"""
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
        print(f"   ⚠️ Diretório de validação não encontrado. Tentando: {val_dirs}")
        print(f"   ℹ️ Certifique-se de que o dataset está em: {cfg['path']}")
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
    
    # Matriz de confusão (Linhas: GT, Colunas: Pred)
    # Última linha/coluna = Background (FN/FP)
    matrix = np.zeros((N_CLASSES + 1, N_CLASSES + 1), dtype=int)
    
    all_gt_classes = []
    all_pred_classes = []
    all_scores = []
    
    tp_per_class = np.zeros(N_CLASSES)
    fp_per_class = np.zeros(N_CLASSES)
    fn_per_class = np.zeros(N_CLASSES)
    
    for img_file in tqdm(img_files, desc=f"   Validando {model_name}", leave=False):
        # Carrega ground truth
        img = Image.open(img_file)
        img_w, img_h = img.size
        label_file = img_file.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        gt_boxes, gt_classes = load_ground_truth(label_file, img_w, img_h)
        
        # Obtém predições
        pred_boxes, pred_classes, pred_scores = get_predictions(model, model_type, img_file, conf_threshold)
        
        matched_gt = set()
        matched_pred = set()
        
        # Matching com IoU
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
                    
                    if g_cls == p_cls:
                        tp_per_class[g_cls] += 1
                    else:
                        fp_per_class[p_cls] += 1
                        fn_per_class[g_cls] += 1
                    
                    all_gt_classes.append(g_cls)
                    all_pred_classes.append(p_cls)
                    all_scores.append(pred_scores[best_p_idx])
        
        # False Negatives (GT não matched)
        for g_idx, g_cls in enumerate(gt_classes):
            if g_idx not in matched_gt:
                matrix[g_cls, N_CLASSES] += 1
                fn_per_class[g_cls] += 1
        
        # False Positives (Pred não matched)
        for p_idx, p_cls in enumerate(pred_classes):
            if p_idx not in matched_pred:
                matrix[N_CLASSES, p_cls] += 1
                fp_per_class[p_cls] += 1
    
    # ==== GERA VISUALIZAÇÕES ====
    
    # 1. Matriz de Confusão
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
    
    # 2. Gráfico de Métricas por Classe
    metrics_per_class = []
    for i, name in enumerate(CLASS_NAMES):
        tp = tp_per_class[i]
        fp = fp_per_class[i]
        fn = fn_per_class[i]
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        metrics_per_class.append({
            'Classe': name,
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4),
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'Support': int(tp + fn)
        })
    
    df_metrics = pd.DataFrame(metrics_per_class)
    
    # Gráfico de barras
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(N_CLASSES)
    width = 0.25
    
    ax1 = axes[0]
    ax1.bar(x - width, df_metrics['Precision'], width, label='Precision', color='steelblue')
    ax1.bar(x, df_metrics['Recall'], width, label='Recall', color='coral')
    ax1.bar(x + width, df_metrics['F1-Score'], width, label='F1-Score', color='seagreen')
    ax1.set_xlabel('Classe')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Métricas por Classe: {model_name}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gráfico de suporte (quantidade de amostras)
    ax2 = axes[1]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    ax2.bar(x - width, df_metrics['TP'], width, label='TP', color=colors[0])
    ax2.bar(x, df_metrics['FP'], width, label='FP', color=colors[1])
    ax2.bar(x + width, df_metrics['FN'], width, label='FN', color=colors[2])
    ax2.set_xlabel('Classe')
    ax2.set_ylabel('Quantidade')
    ax2.set_title('Detecções por Classe')
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = output_dir / f"metricas_{model_name}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"   ✅ Gráfico de métricas salvo: {save_path}")
    
    # Calcula métricas globais
    total_tp = tp_per_class.sum()
    total_fp = fp_per_class.sum()
    total_fn = fn_per_class.sum()
    
    global_precision = total_tp / (total_tp + total_fp + 1e-6)
    global_recall = total_tp / (total_tp + total_fn + 1e-6)
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall + 1e-6)
    
    results = {
        'modelo': model_name,
        'precision_global': round(global_precision, 4),
        'recall_global': round(global_recall, 4),
        'f1_global': round(global_f1, 4),
        'total_tp': int(total_tp),
        'total_fp': int(total_fp),
        'total_fn': int(total_fn),
        'metricas_classe': df_metrics.to_dict('records')
    }
    
    # Salva tabela de métricas
    csv_path = output_dir / f"metricas_{model_name}.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"   ✅ Tabela de métricas salva: {csv_path}")
    
    return results


# ==============================================================================
# 📋 RELATÓRIO COMPARATIVO
# ==============================================================================

def generate_comparison_report(all_results, all_stats, output_dir):
    """Gera relatório comparativo entre todos os modelos"""
    print("\n" + "="*60)
    print("📋 GERANDO RELATÓRIO COMPARATIVO")
    print("="*60)
    
    # Tabela comparativa de avaliação
    if all_results:
        df_comparison = pd.DataFrame([{
            'Modelo': r['modelo'],
            'Precision': r['precision_global'],
            'Recall': r['recall_global'],
            'F1-Score': r['f1_global'],
            'TP': r['total_tp'],
            'FP': r['total_fp'],
            'FN': r['total_fn']
        } for r in all_results if r])
        
        if not df_comparison.empty:
            # Ordena por F1-Score
            df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
            
            print("\n📊 Comparativo de Métricas (ordenado por F1-Score):")
            print(df_comparison.to_string(index=False))
            
            # Salva CSV
            csv_path = output_dir / "comparativo_modelos.csv"
            df_comparison.to_csv(csv_path, index=False)
            
            # Gráfico comparativo
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Gráfico de barras - Métricas
            ax1 = axes[0]
            x = np.arange(len(df_comparison))
            width = 0.25
            ax1.bar(x - width, df_comparison['Precision'], width, label='Precision', color='steelblue')
            ax1.bar(x, df_comparison['Recall'], width, label='Recall', color='coral')
            ax1.bar(x + width, df_comparison['F1-Score'], width, label='F1-Score', color='seagreen')
            ax1.set_xlabel('Modelo')
            ax1.set_ylabel('Score')
            ax1.set_title('Comparativo de Métricas entre Modelos')
            ax1.set_xticks(x)
            ax1.set_xticklabels(df_comparison['Modelo'], rotation=45, ha='right')
            ax1.legend()
            ax1.set_ylim(0, 1.1)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Gráfico de barras - TP/FP/FN
            ax2 = axes[1]
            ax2.bar(x - width, df_comparison['TP'], width, label='TP', color='#2ecc71')
            ax2.bar(x, df_comparison['FP'], width, label='FP', color='#e74c3c')
            ax2.bar(x + width, df_comparison['FN'], width, label='FN', color='#f39c12')
            ax2.set_xlabel('Modelo')
            ax2.set_ylabel('Quantidade')
            ax2.set_title('Total de Detecções por Modelo')
            ax2.set_xticks(x)
            ax2.set_xticklabels(df_comparison['Modelo'], rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            save_path = output_dir / "comparativo_modelos.png"
            plt.savefig(save_path)
            plt.close()
            print(f"\n✅ Gráfico comparativo salvo: {save_path}")
    
    # Tabela de histórico de treinamento
    if all_stats:
        valid_stats = [s for s in all_stats if s]
        if valid_stats:
            df_stats = pd.DataFrame(valid_stats)
            print("\n📈 Histórico de Treinamento:")
            print(df_stats.to_string(index=False))
            
            csv_path = output_dir / "historico_treinamento.csv"
            df_stats.to_csv(csv_path, index=False)


# ==============================================================================
# 🚀 FUNÇÃO PRINCIPAL
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("🔬 AVALIAÇÃO COMPLETA - ABORDAGEM 2")
    print("="*60)
    print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Diretório de saída: {OUTPUT_DIR}")
    
    all_results = []
    all_stats = []
    
    # Define os modelos a avaliar
    models_config = [
        # YOLOv5
        {
            'weights': ABORDAGEM_DIR / 'yolo5' / 'yolo5_best_abordagem_2_3k.pt',
            'csv': ABORDAGEM_DIR / 'yolo5' / 'yolo5_results_abordagem_2_3k.csv' if (ABORDAGEM_DIR / 'yolo5' / 'yolo5_results_abordagem_2_3k.csv').exists() else None,
            'type': 'yolo',
            'dataset': '3k',
            'name': 'YOLOv5_3k'
        },
        {
            'weights': ABORDAGEM_DIR / 'yolo5' / 'yolo5_best_abordagem_2_7k.pt',
            'csv': ABORDAGEM_DIR / 'yolo5' / 'yolo5_results_abordagem_2_7k.csv' if (ABORDAGEM_DIR / 'yolo5' / 'yolo5_results_abordagem_2_7k.csv').exists() else None,
            'type': 'yolo',
            'dataset': '7k',
            'name': 'YOLOv5_7k'
        },
        # YOLO11
        {
            'weights': ABORDAGEM_DIR / 'yolo11' / 'yolo11_best_abordagem_2_3k.pt',
            'csv': ABORDAGEM_DIR / 'yolo11' / 'yolo11_results_abordagem_2_3k.csv',
            'type': 'yolo',
            'dataset': '3k',
            'name': 'YOLO11_3k'
        },
        {
            'weights': ABORDAGEM_DIR / 'yolo11' / 'yolo11_best_abordagem_2_7k.pt',
            'csv': ABORDAGEM_DIR / 'yolo11' / 'yolo11_results_abordagem_2_7k.csv',
            'type': 'yolo',
            'dataset': '7k',
            'name': 'YOLO11_7k'
        },
        # Faster R-CNN
        {
            'weights': ABORDAGEM_DIR / 'faster' / 'faster_best_abordagem_2_3k.pth',
            'csv': ABORDAGEM_DIR / 'faster' / 'faster_results_abordagem_2_3k.csv',
            'type': 'faster',
            'dataset': '3k',
            'name': 'FasterRCNN_3k'
        },
        {
            'weights': ABORDAGEM_DIR / 'faster' / 'faster_best_abordagem_2_7k.pth',
            'csv': ABORDAGEM_DIR / 'faster' / 'faster_results_abordagem_2_7k.csv',
            'type': 'faster',
            'dataset': '7k',
            'name': 'FasterRCNN_7k'
        },
    ]
    
    # Processa cada modelo
    for config in models_config:
        weights_path = config['weights']
        csv_path = config.get('csv')
        model_type = config['type']
        dataset = config['dataset']
        model_name = config['name']
        
        # Verifica se arquivo de pesos existe
        if not weights_path.exists():
            print(f"\n⚠️ Pesos não encontrados: {weights_path}")
            continue
        
        # Define YAML do dataset
        yaml_path = PROJECT_ROOT / f"data_{dataset}.yaml"
        
        # 1. Analisa histórico de treinamento (CSV)
        if csv_path and Path(csv_path).exists():
            stats = analyze_training_history(csv_path, model_name, OUTPUT_DIR)
            all_stats.append(stats)
        
        # 2. Avalia modelo (matriz de confusão e métricas)
        results = evaluate_model(
            weights_path=str(weights_path),
            model_type=model_type,
            dataset_yaml=str(yaml_path),
            model_name=model_name,
            output_dir=OUTPUT_DIR
        )
        all_results.append(results)
    
    # 3. Gera relatório comparativo
    generate_comparison_report(all_results, all_stats, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("✅ AVALIAÇÃO CONCLUÍDA!")
    print(f"📁 Resultados salvos em: {OUTPUT_DIR}")
    print("="*60)
    
    return all_results, all_stats


if __name__ == "__main__":
    results, stats = main()

