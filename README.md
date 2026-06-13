<div align="center">

# 🔬 Deepfake Dashboard

### Motor de Deteção Forense v2.0 — Sistema de Plugins

Uma plataforma de análise forense de deepfakes que combina **6 detetores visuais** e **5 analisadores ao nível do vídeo** (visual, áudio, metadados, temporal e fisiológico) para detetar manipulações em vídeos e imagens, apresentando os resultados num painel interativo com exportação de relatórios PDF.

[![Python](https://img.shields.io/badge/Python-3.10–3.11-blue?logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 📋 Índice

- [Arquitetura](#-arquitetura)
- [Plugins & Modelos](#-plugins--modelos)
- [Funcionalidades](#-funcionalidades)
- [Setup Rápido](#-setup-rápido)
- [Uso](#-uso)
- [Batch Analyzer](#-batch-analyzer)
- [Testes & Benchmarks](#-testes--benchmarks)
- [Stack Tecnológica](#-stack-tecnológica)
- [Referências Académicas](#-referências-académicas)

---

## 🏗 Arquitetura

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Frontend (Next.js)                         │
│  ┌──────────┐   ┌──────────────────┐   ┌──────────────────────────┐ │
│  │  Upload   │   │ ReportDashboard  │   │ VideoForensicsPlayer     │ │
│  │  + D&D    │   │ Scores + Charts  │   │ Frame viewer + BBoxes    │ │
│  └──────────┘   └──────────────────┘   └──────────────────────────┘ │
│         │                ▲                          ▲               │
│         │     Progress   │         Frame Cache      │               │
│         │     Polling    │         /api/frame        │               │
│         ▼                │                          │               │
├──────────────────────────────────────────────────────────────────────┤
│                     POST /api/analyze                                │
├──────────────────────────────────────────────────────────────────────┤
│                        Backend (FastAPI)                             │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                      PluginManager                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │  │
│  │  │ FacePreProc  │  │ SceneClassif │  │ Auto-discovery        │ │  │
│  │  │ MTCNN + SSD  │  │ 3 tipos cena │  │ plugins/*.py          │ │  │
│  │  └──────────────┘  └──────────────┘  └───────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  6 PLUGINS VISUAIS (por frame, com routing por cena)                  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐ │
│  │MesoNet │ │  ViT   │ │  DCT   │ │  Edge  │ │  PRNU  │ │Sightengin│ │
│  │Meso-4  │ │99.27%¹ │ │1/f²law │ │ Blend  │ │ noise  │ │ e (cloud)│ │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └──────────┘ │
│  5 ANALISADORES AO NÍVEL DO VÍDEO (sobre o vídeo inteiro)             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │ Lip Sync │ │WavLM áudio│ │ Metadata │ │ Temporal │ │   rPPG     │ │
│  │FaceMesh  │ │voz sintét.│ │EXIF/probe│ │ coerência│ │ pulso card.│ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Fluxo de Análise

1. **Upload** — O utilizador arrasta um vídeo/imagem; `POST /api/analyze` aceita o ficheiro, **enfileira** o trabalho em background e devolve o `task_id` imediatamente (worker FastAPI não bloqueia)
2. **Extração** — FFmpeg extrai frames a 2fps (ou OpenCV como fallback)
3. **Pré-processamento** — MTCNN deteta faces, SceneClassifier classifica a cena
4. **Análise Visual** — Os 6 plugins processam cada frame com routing por cena (plugins face-condicionais são saltados em cenas sem rosto). ViT e MesoNet suportam batching quando há ≥2 faces num frame
5. **Análise ao nível do vídeo** — 5 analisadores correm sobre o vídeo inteiro: WavLM analisa o áudio em **chunks de 10s** (até 3 min); MediaPipe correlaciona áudio/lábios em **janelas deslizantes de 4s** (até 2 min); e ainda metadados (EXIF/ffprobe), coerência temporal de landmarks e rPPG (pulsação via canal verde)
6. **Scoring** — Agregação por cena (média ponderada) → MAX sobre faces → média sobre frames; veredito final por MAX(visual, áudio). O frontend classifica em 4 categorias (sem evidência / inconclusivo / consistente com manipulação / manipulação altamente provável) em vez de binário
7. **Polling** — Frontend faz polling de `/api/progress`; ao detetar `stage=done` busca `/api/result/{task_id}`
8. **Dashboard** — Resultados apresentados com gráficos, timeline, bounding boxes, e métricas separadas

---

## 🔌 Plugins & Modelos

### 6 plugins visuais (por frame)

| Plugin | Técnica | Modelo/Base |
|--------|---------|-------------|
| **MesoNet** | CNN compacta (~30K params) purpose-trained para deteção de face-swap; opera na escala mesoscópica | Meso-4 (Afchar et al., WIFS 2018), pesos Keras convertidos para PyTorch |
| **ViT Detector** | Vision Transformer fine-tuned em 140K+ imagens fake/real | `dima806/deepfake_vs_real_image_detection` (99.27% in-distribution¹) |
| **DCT Frequency Analyzer** | Análise espectral FFT — desvio da lei 1/f², artefactos de stride, spectral flatness | Baseado em física (sem ML) |
| **Edge Blending Detector** | Descontinuidades de gradiente, shifts de cor HSV, inconsistências de iluminação na fronteira da face | Inspirado em *Face X-ray* (Li et al., CVPR 2020) |
| **PRNU Noise Detector** | Variância de ruído residual com high-pass Gaussiano (proxy single-frame do PRNU clássico) | Baseado em física (sem ML) |
| **Sightengine Cloud** | API cloud comercial com deteção de AI-generated content (opcional, rate-limited) | Sightengine API |

> **¹ Sobre o 99.27%**: é a accuracy reportada pelo autor do modelo no **dataset de treino**. Em dados out-of-distribution (vídeos comprimidos, deepfakes de métodos novos, demografias sub-representadas) o número cai. Ver [`BENCHMARKS.md`](BENCHMARKS.md) para resultados medidos em datasets públicos.

### 5 analisadores ao nível do vídeo

| Analisador | Técnica | Modelo/Base |
|------------|---------|-------------|
| **Lip Sync** | Correlação entre energia áudio (STFT) e abertura labial via FaceMesh 3D em janelas deslizantes de 4s | MediaPipe FaceMesh (478 landmarks) |
| **WavLM Audio** | Deteção de voz sintética (clonagem de voz AI) — chunks de 10s ao longo do vídeo | `abhishtagatya/wavlm-base-960h-itw-deepfake` |
| **Metadata** | Inspeção de EXIF (imagens) e ffprobe (vídeos) para inconsistências de codec/encoding | Heurístico |
| **Temporal Coherence** | Instabilidade temporal de landmarks faciais entre frames consecutivos | Heurístico |
| **rPPG** | Deteção de pulsação cardíaca via oscilação periódica do canal verde do rosto | Inspirado em *FakeCatcher* (Ciftci et al., 2020) |

### Routing por cena

Os pesos por cena estão definidos em [`engine/core/scene_classifier.py`](engine/core/scene_classifier.py) e somam 1.0 por cena. Plugins não listados para uma cena são **saltados** (não executados).

| Cena | Detetada quando… | Plugins ativos (peso) |
|---|---|---|
| `CROPPED_FACE` | Face ≥ 50% do frame | MesoNet 0.40 · Sightengine 0.20 · ViT 0.18 · DCT 0.10 · Edge 0.07 · PRNU 0.05 |
| `FACE_IN_SCENE` | Face detetada mas < 50% do frame | MesoNet 0.38 · Sightengine 0.18 · ViT 0.18 · DCT 0.10 · Edge 0.09 · PRNU 0.07 |
| `NO_FACE` | Nenhuma face detetada | DCT 0.50 · Sightengine 0.50 (MesoNet, ViT, Edge e PRNU **excluídos**: precisam de uma face) |

---

## ✨ Funcionalidades

- 🎯 **Análise multi-face** — Deteta e analisa até 6 caras por frame; caras < 3% do frame são mostradas mas **não influenciam o veredito** (evitam falsos positivos por scores ruidosos em faces pequenas)
- 🧠 **6 plugins visuais** complementares (ML + física + cloud) com auto-discovery
- 🎬 **5 analisadores de vídeo** — lip-sync, voz sintética (WavLM), metadados, coerência temporal e rPPG, cobrindo o vídeo inteiro
- 📊 **Dashboard interativo** — Gráfico temporal, timeline de frames, bounding boxes com canvas overlay
- 📄 **Exportação PDF** — Relatório forense completo com tabelas e métricas
- 🗃️ **Histórico persistente** — Cada análise é guardada em SQLite local (`~/.deepfake-forensics/history.db`), com vista dedicada (`/historico`) e endpoints REST CRUD
- ⏳ **Endpoint assíncrono** — `POST /api/analyze` enfileira e devolve `task_id`; worker FastAPI fica livre para outros pedidos enquanto a análise corre
- 🚦 **4 categorias de veredito** — sem evidência / inconclusivo / consistente com manipulação / manipulação altamente provável (não apenas binário)
- 🌗 **Dark/Light mode** — Toggle com persistência em localStorage
- 🔄 **Batch Analyzer** — CLI multi-threaded com confusion matrix, F1-Score, MCC
- 📈 **Benchmark script** — `engine/benchmark.py` corre o motor sobre datasets rotulados e emite AUC/F1 por plugin, com intervalos de confiança a 95% via bootstrap
- 🎯 **Scene-aware routing** — Plugins são ativados apenas quando relevantes (CROPPED_FACE, FACE_IN_SCENE, NO_FACE)
- ⚡ **Worst-case scoring** — `MAX(visual, audio)` garante que nenhuma manipulação escapa
- 🔒 **Auth + rate-limit + upload cap** — `X-API-Key` opcional, 10 req/min por cliente, upload ≤ 200 MB (configuráveis)
- ✅ **Suite de testes** — 82 unit tests cobrem scene routing, DCT, MesoNet, lip sync logic, ciclo de vida dos plugins, history store e os bugs de regressão críticos

---

## 🚀 Setup Rápido

### Pré-requisitos

- **Python 3.10+** com pip
- **Node.js 18+** com npm
- **Git**

### 1. Clonar o repositório

```bash
git clone https://github.com/Americo05/DFForensics.git
cd DFForensics
```

### 2. Backend (FastAPI + Python)

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Instalar dependências
pip install fastapi uvicorn opencv-python-headless numpy
pip install torch torchvision transformers pillow   # ViT + WavLM
pip install mediapipe                                # Lip Sync
pip install librosa imageio-ffmpeg                   # Áudio
pip install facenet-pytorch                          # MTCNN
pip install requests python-dotenv python-multipart  # API + uploads

# Iniciar o servidor
cd engine
uvicorn main:app --port 8000
```

### 3. Frontend (Next.js)

```bash
# Numa nova terminal, na raiz do projeto
npm install
npm run dev
```

### 4. Abrir o Dashboard

Acede a **http://localhost:3000** no browser.

### Configuração do Sightengine (opcional)

Para ativar o plugin Cloud, cria um ficheiro `.env` na pasta `engine/`:

```env
SIGHTENGINE_API_USER=<teu_user>
SIGHTENGINE_API_SECRET=<teu_secret>
```

### Configuração de segurança (recomendada para produção)

| Variável | Default | Descrição |
|---|---|---|
| `ENGINE_API_KEYS` | _(vazia → auth DESLIGADA)_ | Lista separada por vírgulas. Quando definida, todas as rotas exigem header `X-API-Key`. |
| `ENGINE_RATE_LIMIT_PER_MIN` | `10` | Pedidos por minuto, por cliente (API key ou IP). |
| `ENGINE_MAX_UPLOAD_MB` | `200` | Tamanho máximo do upload em megabytes. |

No frontend, define `NEXT_PUBLIC_ENGINE_API_KEY` em `.env.local` para o browser enviar o header automaticamente:

```env
NEXT_PUBLIC_ENGINE_API_KEY=<mesma chave do backend>
```

> **Sem `ENGINE_API_KEYS` o motor aceita qualquer pedido** — está OK para desenvolvimento local mas **não exponhas publicamente** nesse modo.

---

## 📖 Uso

1. **Arrasta** um vídeo ou imagem para a zona de upload
2. (Opcional) Ativa/desativa a **Cloud API** com o toggle
3. Clica em **"Iniciar Análise Forense"**
4. Acompanha o **progresso em tempo real** na barra
5. Explora os resultados no **dashboard interativo**:
   - Veredito separado (Visual / Áudio)
   - Gráfico temporal de suspeição
   - Visualizador frame-a-frame com bounding boxes
   - Detalhes por plugin
6. Clica em **"Exportar Relatório PDF"** para guardar o relatório

---

## 📦 Batch Analyzer

Para testar em lote contra datasets:

```bash
cd engine
python batch_analyze.py "C:\path\to\dataset" --label FAKE --limit 100

# Multi-servidor
python batch_analyze.py "C:\datasets\faces" --api "http://localhost:8000,http://192.168.1.2:8000" --label REAL
```

**Output:** CSV com scores por imagem, confusion matrix, Precision, Recall, F1-Score e MCC.

Tecla **P** para pausar/retomar o lote durante a execução.

---

## 🧪 Testes & Benchmarks

### Testes unitários

```bash
.venv/Scripts/python.exe -m pytest engine/tests -v
```

Cobertura atual: **82 testes** sobre `SceneClassifier` (geometria + integridade das tabelas de pesos), `DCTFrequencyAnalyzer` (incluindo regressão da regressão linear), `MesoNetDetector` (carregamento de pesos + flatten order), `LipSyncAnalyzer` (correlação áudio↔boca em vários cenários), ciclo de vida dos plugins (reset, propagação de `face_roi=None`, filtro multi-face), os analisadores de vídeo, e o `history_store` (SQLite CRUD).

### Benchmark de accuracy em dataset rotulado

```bash
# Estrutura esperada:
# datasets/
#   real/  *.mp4, *.jpg
#   fake/  *.mp4, *.jpg

cd engine
python benchmark.py --dataset ../datasets --limit 100
```

Gera CSV com scores por ficheiro + `BENCHMARKS.md` com AUC, F1, precision e recall **por plugin** e **globais**. Datasets sugeridos: [FaceForensics++](https://github.com/ondyari/FaceForensics), [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics).

Resultados publicados (quando disponíveis) em [`BENCHMARKS.md`](BENCHMARKS.md).

---

## 🛠 Stack Tecnológica

### Backend
| Tecnologia | Versão | Função |
|------------|--------|--------|
| Python | 3.10+ | Linguagem principal |
| FastAPI | 0.136 | API REST |
| PyTorch | 2.11 | Inferência ViT + WavLM |
| Transformers | 5.6 | Modelos HuggingFace |
| MediaPipe | 0.10.14 | FaceMesh 3D (Lip Sync) |
| OpenCV | 4.13 | Processamento de imagem |
| Librosa | 0.11 | Análise de sinal áudio |
| MTCNN (facenet-pytorch) | 2.6 | Deteção facial |

### Frontend
| Tecnologia | Versão | Função |
|------------|--------|--------|
| Next.js | 16 | Framework React |
| React | 19.2 | UI Components |
| Tailwind CSS | 4 | Styling |
| Recharts | 3.8 | Gráficos temporais |
| Lucide React | 0.577 | Ícones |
| jsPDF | — | Exportação PDF |

---

## 📚 Referências Académicas

1. **MesoNet: a Compact Facial Video Forgery Detection Network** — Afchar, D. et al., IEEE WIFS 2018
   - *Arquitetura Meso-4 usada no plugin principal de deteção visual*

2. **Face X-ray for More General Face Forgery Detection** — Li, L. et al., CVPR 2020
   - *Base teórica para o plugin Edge Blending Boundary Detector*

3. **FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals** — Ciftci, U. A. et al., IEEE TPAMI 2020
   - *Base teórica para o analisador rPPG (pulsação cardíaca)*

4. **WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing** — Chen, S. et al., IEEE JSTSP 2022
   - *Modelo usado para deteção de clonagem de voz (WavLM-base-960h)*

5. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** — Dosovitskiy, A. et al., ICLR 2021
   - *Arquitetura ViT usada num dos plugins de deteção visual*

6. **MediaPipe Face Mesh** — Kartynnik, Y. et al., Google Research, 2019
   - *478 pontos faciais 3D usados para análise de sincronia labial*

7. **Modelling the Power Spectra of Natural Images: Statistics and Information** — van der Schaaf, A. & van Hateren, J.H., Vision Research, 1996
   - *Fundamentação da lei 1/f² usada no plugin DCT Frequency Analyzer*

8. **FaceForensics++: Learning to Detect Manipulated Facial Images** — Rössler, A. et al., ICCV 2019
   - *Dataset de referência para validação do sistema de deteção*

---

<div align="center">

**Deepfake Dashboard** — Projeto Académico · UTAD · 2025/2026

</div>
