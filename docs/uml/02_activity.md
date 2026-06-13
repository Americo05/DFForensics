# Diagrama de Atividade — Fluxo principal de análise

Mapeia o fluxo de trabalho do caso de uso **UC1 (Analisar Mídia)**, desde o upload do utilizador até à disponibilização do resultado (Cap. 3.3 do relatório).

```mermaid
%%{init: {'flowchart': {'curve': 'basis', 'padding': 20, 'nodeSpacing': 50, 'rankSpacing': 60}, 'themeVariables': {'fontSize': '28px', 'fontFamily': 'Inter, Arial, sans-serif'}}}%%
flowchart TB
    Start([Utilizador])
    Upload[Selecciona ficheiro<br/>POST /api/analyze]

    subgraph Validacao["Validação síncrona"]
        direction LR
        Auth{Auth OK?}
        Rate{Rate limit?}
        Size{Size ≤ cap?}
        Auth -- sim --> Rate
        Rate -- não --> Size
    end

    Reject[/HTTP 4xx<br/>401 / 429 / 413/]

    SaveTmp[Grava em /tmp<br/>cria task_id]
    Respond[/HTTP 202<br/>task_id/]

    SubscribeSSE[Cliente subscreve SSE]
    BgStart[Background Task arranca]

    Extract[Extrai frames<br/>cv2.VideoCapture]
    LoopFrames{Próximo frame?}

    subgraph FrameLoop["Para cada frame"]
        direction LR
        FaceDetect[MTCNN<br/>face detect]
        SceneClass[SceneClassifier]
        BatchPath{Batch<br/>elegível?}
        BatchInfer[analyze_frames_batch<br/>ViT + MesoNet]
        PerFace[analyze_frame<br/>plugins ativos]
        Aggregate[MAX faces<br/>por frame]
        EmitSSE[Emit SSE]
        FaceDetect --> SceneClass --> BatchPath
        BatchPath -- sim --> BatchInfer --> PerFace
        BatchPath -- não --> PerFace
        PerFace --> Aggregate --> EmitSSE
    end

    subgraph PosAnalise["Pós-análise vídeo-level"]
        direction LR
        VideoAnalyzers[Metadata · Temporal<br/>rPPG · LipSync · Audio]
        FinalScore[overall_score]
        Persist[Persiste<br/>memória + SQLite]
        VideoAnalyzers --> FinalScore --> Persist
    end

    NotifyDone[SSE: completed]

    subgraph Cliente["Cliente"]
        direction LR
        Inspect[Inspeciona frames<br/>+ bboxes]
        OptPDF{Gerar PDF?}
        PDF[Descarrega PDF]
        Inspect --> OptPDF
        OptPDF -- sim --> PDF
    end

    End([Fim])

    Start --> Upload --> Validacao
    Validacao -- falha --> Reject
    Validacao -- sucesso --> SaveTmp
    SaveTmp --> Respond
    Respond --> SubscribeSSE
    Respond --> BgStart

    BgStart --> Extract --> LoopFrames
    LoopFrames -- sim --> FrameLoop
    FrameLoop --> LoopFrames
    LoopFrames -- não --> PosAnalise
    PosAnalise --> NotifyDone

    SubscribeSSE --> NotifyDone
    NotifyDone --> Cliente
    Cliente --> End
    OptPDF -- não --> End

    classDef startend fill:#a3be8c,stroke:#2e3440,color:#2e3440
    classDef decision fill:#ebcb8b,stroke:#2e3440,color:#2e3440
    classDef error fill:#bf616a,stroke:#2e3440,color:#eceff4
    classDef action fill:#5e81ac,stroke:#2e3440,color:#eceff4
    classDef async fill:#b48ead,stroke:#2e3440,color:#eceff4

    class Start,End startend
    class Auth,Rate,Size,LoopFrames,BatchPath,OptPDF decision
    class Reject error
    class Upload,SaveTmp,Respond,Extract,FaceDetect,SceneClass,PerFace,Aggregate,VideoAnalyzers,FinalScore,Persist,Inspect,PDF action
    class BgStart,BatchInfer,EmitSSE,NotifyDone,SubscribeSSE async
```

## Notas de leitura

### Caminhos paralelos
Após `Respond` (HTTP 202), o fluxo bifurca em duas atividades concorrentes:
- **Cliente**: subscreve SSE e aguarda eventos.
- **Servidor**: executa análise em background thread (FastAPI `BackgroundTasks`).

A reconvergência acontece em `NotifyDone`, quando o servidor emite o evento final via SSE.

### Pontos de decisão críticos
| Decisão | Lógica | Localização |
|---------|--------|-------------|
| `SUPPORTS_BATCH` | Atributo de classe nos plugins ([linha 145 mesonet_detector.py](../../engine/plugins/mesonet_detector.py#L145), [linha 52 vit_detector.py](../../engine/plugins/vit_detector.py#L52)) | `plugin_manager._run_analysis_locked` |
| Cenário (CROPPED_FACE / FACE_IN_SCENE / NO_FACE) | Rácio área da face / área do frame ≥ 0.50 | `SceneClassifier.classify` |
| Plugin ativo | Presença em `SCENE_PLUGIN_WEIGHTS[scene]` | `SceneClassifier.get_active_plugins_and_weights` |

### Optimizações representadas no diagrama
1. **Face detection partilhada**: MTCNN corre uma vez por frame, resultado reutilizado por todos os plugins (evita N invocações).
2. **Batched inference**: quando ≥2 caras no mesmo frame e o plugin suporta batch, processa todas numa só forward pass (~5× speedup em CPU, mais em GPU).
3. **Skip de plugins irrelevantes**: cena `NO_FACE` salta MesoNet, ViT, Edge Blending, PRNU (que requerem ROI facial).
