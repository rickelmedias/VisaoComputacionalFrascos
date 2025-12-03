# 📊 Análise e Evolução - Abordagem 2

**Data da Análise:** 03/12/2025  
**Projeto:** Detecção de Frascos com Visão Computacional  
**Modelos Avaliados:** YOLOv5, YOLO11, Faster R-CNN

---

## 📋 Resumo Executivo

Este documento apresenta uma análise detalhada dos modelos treinados na **Abordagem 2** para detecção de frascos em 4 classes: `Danificado`, `Frasco`, `Lacrado` e `Sem_lacre`.

### 🏆 Ranking de Desempenho (mAP@50-95)

| Posição | Modelo | Dataset | mAP@50-95 | mAP@50 | Épocas |
|:-------:|--------|---------|-----------|--------|--------|
| 🥇 1º | **YOLO11** | 7k | **97.30%** | 99.34% | 178 |
| 🥈 2º | YOLOv5 | 7k | 97.04% | 99.31% | 127 |
| 🥉 3º | Faster R-CNN | 7k | 96.06% | 98.72% | 30 |
| 4º | YOLOv5 | 3k | 90.72% | 97.87% | 288 |
| 5º | Faster R-CNN | 3k | 90.00% | 98.21% | 42 |
| 6º | YOLO11 | 3k | 89.91% | 97.58% | 131 |

---

## 📈 Análise Detalhada por Modelo

### 1. YOLOv5

#### Dataset 3k
- **Épocas totais:** 288 (early stopping)
- **Melhor mAP@50-95:** 90.72%
- **Melhor mAP@50:** 97.87%
- **Loss final (treino):** 0.351
- **Loss final (validação):** 0.347

**Observações:**
- Treinou por muitas épocas (288) antes do early stopping
- Boa convergência, mas o dataset menor limitou o desempenho
- Loss de treino e validação muito próximas → sem overfitting significativo

#### Dataset 7k
- **Épocas totais:** 127 (early stopping)
- **Melhor mAP@50-95:** 97.04%
- **Melhor mAP@50:** 99.31%
- **Loss final (treino):** 0.259
- **Loss final (validação):** 0.183

**Observações:**
- Convergência mais rápida que no 3k
- Excelente desempenho com mais dados
- Loss de validação menor que treino → modelo generalizando bem

---

### 2. YOLO11

#### Dataset 3k
- **Épocas totais:** 131
- **Melhor mAP@50-95:** 89.91%
- **Melhor mAP@50:** 97.58%
- **Loss final (treino):** 0.435
- **Loss final (validação):** 0.374

**Observações:**
- Desempenho ligeiramente inferior ao YOLOv5 no dataset menor
- Loss mais alta que YOLOv5 → pode indicar dificuldade de convergência com poucos dados

#### Dataset 7k
- **Épocas totais:** 178
- **Melhor mAP@50-95:** 97.30% ⭐ **MELHOR MODELO**
- **Melhor mAP@50:** 99.34%
- **Loss final (treino):** 0.221
- **Loss final (validação):** 0.171

**Observações:**
- **Melhor desempenho geral** entre todos os modelos
- Superou YOLOv5 com dataset maior
- Excelente generalização (loss de val < loss de treino)

---

### 3. Faster R-CNN

#### Dataset 3k
- **Épocas totais:** 42 (early stopping)
- **Melhor mAP@50-95:** 90.00%
- **Melhor mAP@50:** 98.21%
- **Loss final (treino):** 0.045
- **Loss final (validação):** 0.030

**Observações:**
- Convergência muito rápida (42 épocas)
- Loss extremamente baixa → pode indicar capacidade de memorização
- Learning rate schedule agressivo (de 0.005 para 5e-9)

#### Dataset 7k
- **Épocas totais:** 30 (early stopping)
- **Melhor mAP@50-95:** 96.06%
- **Melhor mAP@50:** 98.72%
- **Loss final (treino):** 0.018
- **Loss final (validação):** 0.013

**Observações:**
- Convergência ainda mais rápida
- Excelente desempenho, porém limitado pelo early stopping precoce
- Poderia continuar treinando com learning rate menor

---

## 🔍 Problemas Identificados

### 1. **Early Stopping Muito Agressivo no Faster R-CNN**
O Faster R-CNN está parando muito cedo (30-42 épocas) enquanto ainda poderia melhorar. O scheduler de learning rate está decaindo muito rápido, fazendo com que o modelo estabilize prematuramente.

**Evidência:**
- Loss final extremamente baixa (0.013-0.030)
- Parou antes de atingir o mesmo nível de mAP dos YOLOs

### 2. **Desbalanceamento de Classes**
Pela natureza do problema (detecção de frascos), é provável que existam classes desbalanceadas, especialmente "Danificado" que tende a ser mais raro.

**Impacto esperado:**
- Baixo recall em classes minoritárias
- Bias para classes majoritárias (Frasco, Lacrado)

### 3. **Diferença de Desempenho entre 3k e 7k**
A diferença de ~7% no mAP@50-95 entre datasets indica que:
- Os modelos ainda se beneficiariam de mais dados
- O dataset 3k pode não ter variabilidade suficiente

### 4. **YOLO11 vs YOLOv5 com Poucos Dados**
O YOLO11 performou pior que YOLOv5 no dataset 3k, mas superou no 7k. Isso sugere que:
- YOLO11 requer mais dados para convergir adequadamente
- YOLOv5 é mais robusto para datasets menores

### 5. **Ausência de Métricas por Classe**
Os CSVs não mostram métricas individuais por classe, dificultando identificar problemas específicos como:
- Confusão entre "Lacrado" e "Sem_lacre"
- Dificuldade em detectar "Danificado"

---

## 💡 Recomendações de Melhorias

### 🔧 Curto Prazo (Ajustes de Hiperparâmetros)

#### 1. **Ajustar Scheduler do Faster R-CNN**
```python
# Atual (muito agressivo)
lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# Recomendado (mais suave)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
# ou
lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
```

#### 2. **Aumentar Patience do Early Stopping**
```python
# Atual
PATIENCE = 30

# Recomendado para Faster R-CNN
PATIENCE = 50  # Dar mais tempo para convergir
```

#### 3. **Class Weights para Desbalanceamento**
Para YOLO:
```python
# Adicionar no config.yaml
cls: 1.5  # Aumentar peso da classificação
```

Para Faster R-CNN:
```python
# Usar WeightedRandomSampler (já implementado na abordagem 1)
```

### 📊 Médio Prazo (Melhorias de Arquitetura)

#### 4. **Test-Time Augmentation (TTA)**
Implementar TTA para melhorar predições em tempo de inferência:
```python
# Para YOLO
results = model.predict(img, augment=True)
```

#### 5. **Ensemble de Modelos**
Combinar predições dos 3 modelos para melhor precisão:
- Usar Weighted Box Fusion (WBF)
- YOLO11 como modelo principal, Faster R-CNN para validação

#### 6. **Análise de Erros por Classe**
Implementar matriz de confusão normalizada para identificar:
- Confusões sistemáticas entre classes
- Classes com baixo recall

### 🚀 Longo Prazo (Data e Arquitetura)

#### 7. **Aumento de Dados Específico**
- Mais imagens de "Danificado" (provavelmente subrepresentada)
- Variações de iluminação e ângulo
- Synthetic data generation

#### 8. **Modelos Maiores**
Testar variantes maiores:
- YOLOv5x → YOLOv5x6
- YOLO11x → YOLO11x com imgsz=1280

#### 9. **Fine-tuning em Duas Etapas**
1. Treinar com imgsz=640
2. Fine-tune final com imgsz=1280 e lr baixo

---

## 📁 Arquivos Gerados

```
abordagem_2/avaliacoes/
├── historico_YOLOv5_3k.png      # Gráficos de treinamento YOLOv5 3k
├── historico_YOLOv5_7k.png      # Gráficos de treinamento YOLOv5 7k
├── historico_YOLO11_3k.png      # Gráficos de treinamento YOLO11 3k
├── historico_YOLO11_7k.png      # Gráficos de treinamento YOLO11 7k
├── historico_FasterRCNN_3k.png  # Gráficos de treinamento Faster 3k
├── historico_FasterRCNN_7k.png  # Gráficos de treinamento Faster 7k
└── historico_treinamento.csv    # Tabela comparativa
```

---

## 🎯 Próximos Passos

1. [ ] Baixar datasets 3k e 7k
2. [ ] Executar avaliação completa com matriz de confusão
3. [ ] Implementar ajustes do scheduler no Faster R-CNN
4. [ ] Testar TTA nos modelos YOLO
5. [ ] Analisar erros por classe específica
6. [ ] Considerar ensemble para produção

---

## 📊 Conclusão

O **YOLO11 com dataset 7k** apresentou o melhor desempenho geral (97.30% mAP@50-95), seguido de perto pelo YOLOv5 7k. O Faster R-CNN mostrou convergência rápida mas pode estar sub-otimizado devido ao scheduler agressivo.

**Recomendação para produção:** Usar YOLO11 7k como modelo principal, com possibilidade de ensemble com Faster R-CNN para casos críticos.

---

*Documento gerado automaticamente pelo script de avaliação*

