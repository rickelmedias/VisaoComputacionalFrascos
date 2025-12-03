# 📊 Documentação de Avaliação dos Modelos - Abordagem 2

**Projeto:** Detecção de Frascos com Visão Computacional  
**Data:** Dezembro/2025  
**Modelos Avaliados:** YOLOv5, YOLO11, Faster R-CNN  
**Datasets:** 3k (3.000 imagens de treino) e 7k (7.000 imagens de treino)

---

## 📑 Sumário

1. [Metodologia de Avaliação](#1-metodologia-de-avaliação)
2. [Matriz de Confusão](#2-matriz-de-confusão)
3. [Tabela de Resumo - Métricas e Cálculos](#3-tabela-de-resumo---métricas-e-cálculos)
4. [Histórico Detalhado](#4-histórico-detalhado)
5. [Gráficos de Evolução Comparativa](#5-gráficos-de-evolução-comparativa)
6. [Por que Precision do Faster R-CNN é N/A?](#6-por-que-precision-do-faster-r-cnn-é-na)
7. [Referências Acadêmicas](#7-referências-acadêmicas)

---

## 1. Metodologia de Avaliação

### 1.1 Visão Geral

A avaliação dos modelos foi realizada utilizando o script `avaliar_modelos.py`, que executa as seguintes etapas:

1. **Carregamento dos pesos treinados** (`.pt` para YOLO, `.pth` para Faster R-CNN)
2. **Inferência no conjunto de validação** de cada dataset
3. **Cálculo de métricas** por correspondência de IoU (Intersection over Union)
4. **Geração de visualizações** (matrizes de confusão, gráficos de evolução)

### 1.2 Estrutura dos Datasets

```
dataset/
├── dataset3k/
│   ├── train/images/    # ~3.000 imagens de treino
│   ├── train/labels/    # Anotações YOLO format
│   ├── val/images/      # Imagens de validação
│   └── val/labels/      # Anotações de validação
└── dataset7k/
    ├── train/images/    # ~7.000 imagens de treino
    ├── train/labels/
    ├── val/images/
    └── val/labels/
```

### 1.3 Classes do Dataset

| ID | Classe | Descrição |
|:--:|--------|-----------|
| 0 | Danificado | Frasco com danos visíveis |
| 1 | Frasco | Frasco genérico/normal |
| 2 | Lacrado | Frasco com lacre intacto |
| 3 | Sem_lacre | Frasco sem lacre |

### 1.4 Bibliotecas Utilizadas

| Biblioteca | Versão | Função |
|------------|--------|--------|
| PyTorch | 2.7+ | Framework de deep learning |
| Ultralytics | 8.0+ | Carregamento e inferência YOLO |
| TorchVision | 0.18+ | Faster R-CNN e transforms |
| NumPy | 2.0+ | Cálculos matriciais |
| Pandas | 2.0+ | Manipulação de dados |
| Seaborn/Matplotlib | - | Visualizações |

---

## 2. Matriz de Confusão

### 2.1 O que é a Matriz de Confusão?

A **matriz de confusão** é uma ferramenta fundamental para avaliar o desempenho de modelos de classificação em visão computacional (Padilla et al., 2020). Ela mostra a distribuição de predições corretas e incorretas para cada classe.

### 2.2 Como foi Gerada

Para cada modelo (YOLOv5, YOLO11, Faster R-CNN) nos datasets 3k e 7k:

```python
# Pseudo-código do processo
for imagem in conjunto_validacao:
    # 1. Carrega ground truth (anotações reais)
    gt_boxes, gt_classes = carregar_anotacoes(imagem)
    
    # 2. Executa inferência no modelo
    pred_boxes, pred_classes, scores = modelo.predict(imagem, conf=0.5)
    
    # 3. Matching por IoU (Intersection over Union)
    for cada gt_box:
        encontra pred_box com maior IoU
        se IoU >= 0.5:
            matriz[classe_real, classe_predita] += 1
        senão:
            matriz[classe_real, Background] += 1  # False Negative
    
    # 4. Predições não correspondidas são False Positives
    for pred_box não correspondida:
        matriz[Background, classe_predita] += 1
```

### 2.3 Cálculo do IoU (Intersection over Union)

O IoU é a métrica padrão para avaliar a sobreposição entre caixas delimitadoras (Rezatofighi et al., 2019):

$$IoU = \frac{Área_{Interseção}}{Área_{União}} = \frac{|A \cap B|}{|A \cup B|}$$

Onde:
- $A$ = Caixa predita pelo modelo
- $B$ = Caixa ground truth (anotação real)

**Implementação:**
```python
def box_iou(box1, box2):
    # Calcula coordenadas da interseção
    lt = np.maximum(box1[:, None, :2], box2[:, :2])  # Left-Top
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])  # Right-Bottom
    wh = (rb - lt).clip(min=0)  # Width-Height da interseção
    
    inter = wh[:, :, 0] * wh[:, :, 1]  # Área de interseção
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter  # Área de união
    
    return inter / (union + 1e-6)  # IoU
```

### 2.4 Threshold de Confiança

Utilizamos `conf_threshold = 0.5`, significando que apenas predições com confiança ≥ 50% são consideradas. Este é o threshold padrão utilizado em benchmarks como COCO e Pascal VOC (Lin et al., 2014).

### 2.5 Interpretação da Matriz

```
              Predito
           D    F    L    S   BG
Real  D   [TP]  -    -    -   FN
      F    -   [TP]  -    -   FN
      L    -    -   [TP]  -   FN
      S    -    -    -   [TP] FN
     BG   FP   FP   FP   FP   --
```

- **Diagonal principal** (TP): Predições corretas
- **Última coluna** (FN): Objetos reais não detectados (False Negatives)
- **Última linha** (FP): Detecções falsas (False Positives)

### 2.6 Arquivos Gerados

| Arquivo | Descrição |
|---------|-----------|
| `confusion_matrix_YOLOv5_3k.png` | Matriz para YOLOv5 no dataset 3k |
| `confusion_matrix_YOLOv5_7k.png` | Matriz para YOLOv5 no dataset 7k |
| `confusion_matrix_YOLO11_3k.png` | Matriz para YOLO11 no dataset 3k |
| `confusion_matrix_YOLO11_7k.png` | Matriz para YOLO11 no dataset 7k |
| `confusion_matrix_FasterRCNN_3k.png` | Matriz para Faster R-CNN no dataset 3k |
| `confusion_matrix_FasterRCNN_7k.png` | Matriz para Faster R-CNN no dataset 7k |

---

## 3. Tabela de Resumo - Métricas e Cálculos

### 3.1 Estrutura da Tabela

A tabela `tabela_resumo_modelos.csv` contém as seguintes colunas:

| Coluna | Descrição | Como Calcular |
|--------|-----------|---------------|
| Modelo | Nome do modelo + dataset | - |
| Melhor Época | Época com maior mAP@50-95 | `argmax(mAP@50-95)` |
| mAP@50 | Mean Average Precision @ IoU=0.5 | Ver seção 3.2 |
| mAP@50-95 | Mean Average Precision @ IoU=0.5:0.95 | Ver seção 3.2 |
| Overfitting (val/train) | Razão entre loss de validação e treino | `val_loss / train_loss` |
| Estabilidade (std) | Desvio padrão do mAP nas últimas 10 épocas | `std(mAP[-10:])` |
| Precision | Precisão na melhor época | `TP / (TP + FP)` |
| Recall | Recall na melhor época | `TP / (TP + FN)` |

### 3.2 Cálculo do mAP (Mean Average Precision)

O mAP é a métrica padrão para avaliação de detectores de objetos (Padilla et al., 2020):

#### Passo 1: Calcular Precision e Recall por Classe

$$Precision = \frac{TP}{TP + FP}$$

$$Recall = \frac{TP}{TP + FN}$$

Onde:
- **TP** (True Positive): Detecção correta (IoU ≥ threshold)
- **FP** (False Positive): Detecção incorreta (IoU < threshold ou classe errada)
- **FN** (False Negative): Objeto não detectado

#### Passo 2: Calcular a Curva Precision-Recall

Para cada threshold de confiança, calcula-se Precision e Recall, formando a curva P-R.

#### Passo 3: Calcular AP (Average Precision) por Classe

$$AP = \int_0^1 p(r) \, dr$$

Na prática, usa-se interpolação em 11 pontos ou todos os pontos (método COCO):

$$AP = \sum_{i=1}^{n} (r_i - r_{i-1}) \cdot p_{interp}(r_i)$$

#### Passo 4: Calcular mAP

$$mAP = \frac{1}{N_{classes}} \sum_{i=1}^{N_{classes}} AP_i$$

#### mAP@50 vs mAP@50-95

- **mAP@50**: Calculado com threshold IoU = 0.5 (menos rigoroso)
- **mAP@50-95**: Média do mAP em thresholds IoU de 0.5 a 0.95, em passos de 0.05 (mais rigoroso, padrão COCO)

$$mAP@50\text{-}95 = \frac{1}{10} \sum_{t \in \{0.5, 0.55, ..., 0.95\}} mAP@t$$

### 3.3 Cálculo do Overfitting

O overfitting indica se o modelo está memorizando os dados de treino em vez de generalizar.

**Fórmula:**
$$Overfitting_{ratio} = \frac{Loss_{validação}}{Loss_{treino}}$$

**Interpretação:**
| Valor | Interpretação |
|-------|---------------|
| ≤ 1.0 | Excelente (val_loss < train_loss) |
| 1.0 - 1.2 | Baixo (modelo generalizando bem) |
| 1.2 - 1.5 | Médio (atenção necessária) |
| > 1.5 | Alto (modelo provavelmente com overfitting) |

**Exemplo do dataset:**
```
YOLOv5_7k: val_loss=0.1937, train_loss=0.2726
Ratio = 0.1937 / 0.2726 = 0.7106 → ✅ Excelente (val < train)
```

### 3.4 Cálculo da Estabilidade

A estabilidade mede a consistência do modelo nas últimas épocas de treinamento.

**Fórmula:**
$$Estabilidade = \sigma(mAP_{últimas\,10\,épocas})$$

Onde $\sigma$ é o desvio padrão.

**Interpretação:**
| Valor | Interpretação |
|-------|---------------|
| ≤ 0.01 | Alta estabilidade (convergência suave) |
| 0.01 - 0.03 | Média estabilidade |
| > 0.03 | Baixa estabilidade (oscilações) |

**Exemplo:**
```
YOLO11_7k: std = 0.00004 → ⭐ Alta estabilidade
YOLOv5_3k: std = 0.001659 → ⭐ Alta estabilidade
```

---

## 4. Histórico Detalhado

### 4.1 Estrutura do `historico_detalhado.csv`

| Coluna | Descrição |
|--------|-----------|
| modelo | Nome do modelo |
| epochs | Total de épocas treinadas |
| best_epoch | Época com melhor performance |
| best_map50_95 | Melhor mAP@50-95 alcançado |
| best_map50 | Melhor mAP@50 alcançado |
| precision | Precision na melhor época |
| recall | Recall na melhor época |
| best_train_loss | Loss de treino na melhor época |
| best_val_loss | Loss de validação na melhor época |
| overfitting_value | Diferença (val_loss - train_loss) |
| overfitting_ratio | Razão (val_loss / train_loss) |
| overfitting_class | Classificação do overfitting |
| stability_std | Desvio padrão do mAP |
| stability_class | Classificação da estabilidade |

### 4.2 Análise dos Valores

```
Modelo          | Overfitting Ratio | Estabilidade (std)
----------------|-------------------|-------------------
YOLO11_7k       | 0.7710 ✅         | 0.000040 ⭐
YOLOv5_7k       | 0.7106 ✅         | 0.000437 ⭐
FasterRCNN_7k   | 0.7354 ✅         | 0.000005 ⭐
YOLOv5_3k       | 0.9435 ✅         | 0.001659 ⭐
FasterRCNN_3k   | 0.6609 ✅         | N/A
YOLO11_3k       | 0.8479 ✅         | 0.004497 ⭐
```

**Observações:**
- Todos os modelos apresentam **overfitting baixo** (ratio < 1.0), indicando excelente generalização
- A **estabilidade é alta** em todos os modelos, com desvio padrão < 0.01
- Modelos 7k têm melhor estabilidade que os 3k (mais dados = convergência mais suave)

---

## 5. Gráficos de Evolução Comparativa

### 5.1 Gráficos de Histórico Individual

Para cada modelo, geramos um gráfico 2x2 com:

1. **Evolução do mAP** (mAP@50 e mAP@50-95 ao longo das épocas)
2. **Loss de Treinamento** (box_loss, cls_loss, dfl_loss)
3. **Loss de Validação** (mesmas componentes)
4. **Precision e Recall** (evolução ao longo do treino)

**Importância:**
- Identificar se o modelo convergiu ou ainda estava melhorando
- Detectar sinais de overfitting (divergência entre train e val loss)
- Avaliar estabilidade visual do treinamento

### 5.2 Gráficos Comparativos (evolucao_comparativa_*.png)

Comparam os 3 modelos lado a lado:

**Gráfico Esquerdo:** Evolução do mAP@50-95
- Permite visualizar qual modelo convergiu mais rápido
- Mostra o desempenho relativo ao longo do treinamento

**Gráfico Direito:** Evolução da Loss de Validação
- Menor loss = melhor performance
- Permite identificar qual modelo tem menor erro de predição

### 5.3 Gráfico de Barras Comparativo

O `comparativo_barras_final.png` mostra:
- mAP@50-95
- mAP@50
- Precision
- Recall

Para cada modelo, facilitando comparação direta.

---

## 6. Por que Precision do Faster R-CNN é N/A?

### 6.1 Explicação Técnica

O Faster R-CNN utiliza um formato de logging diferente dos modelos YOLO durante o treinamento:

**YOLO (Ultralytics):**
```csv
epoch, train/box_loss, ..., metrics/precision(B), metrics/recall(B), metrics/mAP50(B), ...
```

**Faster R-CNN (PyTorch):**
```csv
epoch, train/box_loss, train/cls_loss, ..., mAP50, mAP50-95, ...
```

O script de treinamento do Faster R-CNN **não calcula precision por época** durante o treino, apenas o mAP e o recall através da biblioteca `torchmetrics.MeanAveragePrecision`.

### 6.2 Como Resolver

Para obter a Precision do Faster R-CNN, seria necessário:

1. **Recalcular manualmente** usando o conjunto de validação:
```python
def calculate_precision(model, val_loader):
    tp, fp = 0, 0
    for images, targets in val_loader:
        preds = model(images)
        # Matching por IoU
        # Conta TP e FP
    return tp / (tp + fp)
```

2. **Modificar o script de treino** do Faster R-CNN para registrar precision

### 6.3 O Recall está Disponível

O recall **é** calculado pelo `MeanAveragePrecision` do torchmetrics através do campo `mar_100` (Mean Average Recall @ 100 detections), por isso aparece na tabela.

---

## 7. Referências Acadêmicas

### Métricas de Avaliação

1. **Padilla, R., Netto, S. L., & da Silva, E. A. B. (2020).** A Survey on Performance Metrics for Object-Detection Algorithms. *International Conference on Systems, Signals and Image Processing (IWSSIP)*, 237-242. https://doi.org/10.1109/IWSSIP48289.2020.9145130

2. **Lin, T. Y., et al. (2014).** Microsoft COCO: Common Objects in Context. *European Conference on Computer Vision (ECCV)*, 740-755. https://doi.org/10.1007/978-3-319-10602-1_48

### IoU e Métricas de Localização

3. **Rezatofighi, H., et al. (2019).** Generalized Intersection over Union: A Metric and a Loss for Bounding Box Regression. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 658-666. https://doi.org/10.1109/CVPR.2019.00075

### Arquiteturas de Detecção

4. **Jocher, G., et al. (2022).** YOLOv5 by Ultralytics. Zenodo. https://doi.org/10.5281/zenodo.3908559

5. **Ultralytics (2023).** YOLO11 Documentation. https://docs.ultralytics.com/models/yolo11/

6. **Ren, S., He, K., Girshick, R., & Sun, J. (2017).** Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 39(6), 1137-1149. https://doi.org/10.1109/TPAMI.2016.2577031

### Transfer Learning e Treinamento

7. **Tan, M., & Le, Q. V. (2021).** EfficientNetV2: Smaller Models and Faster Training. *International Conference on Machine Learning (ICML)*, 10096-10106.

### Learning Rate Schedulers

8. **Loshchilov, I., & Hutter, F. (2017).** SGDR: Stochastic Gradient Descent with Warm Restarts. *International Conference on Learning Representations (ICLR)*.

---

## Apêndice: Execução do Script

### Comando para Gerar Avaliações

```bash
conda activate rocm-env-311
cd /path/to/VisaoComputacional-Frascos/abordagem_2
python avaliar_modelos.py
```

### Arquivos de Saída

```
abordagem_2/avaliacoes/
├── confusion_matrix_*.png       # Matrizes de confusão (6 arquivos)
├── historico_*.png              # Gráficos de histórico (6 arquivos)
├── evolucao_comparativa_3k.png  # Comparativo dataset 3k
├── evolucao_comparativa_7k.png  # Comparativo dataset 7k
├── comparativo_barras_final.png # Barras comparativas
├── tabela_resumo_modelos.csv    # Resumo das métricas
└── historico_detalhado.csv      # Histórico completo
```

---

*Documento gerado para fins acadêmicos - Visão Computacional 2025*

