# Diagrama de Classes — Arquitetura do Engine

Representa a estrutura orientada a objetos do backend (Cap. 3.4 do relatório).
Por legibilidade e fiabilidade de render, está dividido em dois diagramas:

1. **Hierarquia de plugins** — contrato `BaseDetectorPlugin` e as 6 implementações concretas.
2. **Orquestração + analyzers vídeo-level** — `PluginManager`, `FacePreProcessor`, `SceneClassifier`, e os analyzers que correm sobre o vídeo completo.

---

## 1. Hierarquia de plugins

```mermaid
classDiagram
    class BaseDetectorPlugin {
        <<abstract>>
        +SUPPORTS_BATCH bool
        +plugin_name() str
        +plugin_description() str
        +plugin_version() str
        +plugin_weight() float
        +analyze_frame(frame, face_roi) float
        +analyze_frames_batch(items) list
        +is_configured() bool
        +reset() None
        +get_plugin_info() dict
    }

    class MesoNetDetectorPlugin {
        +SUPPORTS_BATCH true
        +INPUT_SIZE 256
        -_model
        -_device
        -_weights_path
        +analyze_frame(frame, face_roi) float
        +analyze_frames_batch(items) list
        -_preprocess(img)
    }

    class ViTDetectorPlugin {
        +SUPPORTS_BATCH true
        +RAW_LOW 0.50
        +RAW_HIGH 1.00
        -_pipe
        +analyze_frame(frame, face_roi) float
        +analyze_frames_batch(items) list
        -_calibrate(raw) float
    }

    class DCTFrequencyAnalyzerPlugin {
        +analyze_frame(frame, face_roi) float
        -_power_law_deviation(gray) float
        -_stride_artifact_score(gray) float
        -_spectral_flatness_score(gray) float
    }

    class EdgeBlendingDetector {
        +analyze_frame(frame, face_roi) float
        -_gradient_discontinuity_score(face) float
        -_color_boundary_score(face) float
        -_illumination_consistency_score(face) float
    }

    class PRNUNoiseResidueDetector {
        +MIN_ROI_SIDE 48
        +analyze_frame(frame, face_roi) float
        -_noise_ratio_score(frame, roi) float
        -_locate_roi(frame, roi) tuple
    }

    class SightengineDeepfakeDetector {
        -_frame_counter int
        -_cached_score float
        -_api_configured bool
        +analyze_frame(frame, face_roi) float
        +reset() None
    }

    BaseDetectorPlugin <|-- MesoNetDetectorPlugin
    BaseDetectorPlugin <|-- ViTDetectorPlugin
    BaseDetectorPlugin <|-- DCTFrequencyAnalyzerPlugin
    BaseDetectorPlugin <|-- EdgeBlendingDetector
    BaseDetectorPlugin <|-- PRNUNoiseResidueDetector
    BaseDetectorPlugin <|-- SightengineDeepfakeDetector
```

---

## 2. Orquestração e analyzers vídeo-level

```mermaid
classDiagram
    class PluginManager {
        -_plugins list
        -_preprocessor FacePreProcessor
        -_analysis_lock Lock
        +CLOUD_PLUGIN_NAME str
        +MIN_FACE_AREA_FOR_VERDICT_RATIO 0.03
        +run_analysis(frames, fps, mode, callback) dict
        +get_plugins() list
        -_load_all_plugins() None
        -_run_analysis_locked(args) dict
    }

    class FacePreProcessor {
        -_use_mtcnn bool
        -_mtcnn MTCNN
        -_net Net
        +process(frame) tuple
        -_process_mtcnn(args) tuple
        -_process_ssd(args) tuple
    }

    class SceneClassifier {
        +CROPPED_FACE_RATIO_THRESHOLD 0.50
        +classify(frame, bbox, detected) SceneType
        +get_active_plugins_and_weights(scene, plugins) list
    }

    class SceneType {
        <<enumeration>>
        CROPPED_FACE
        FACE_IN_SCENE
        NO_FACE
    }

    class PluginNames {
        <<constants>>
        +VIT_DETECTOR str
        +DCT_FREQUENCY str
        +EDGE_BLENDING str
        +SIGHTENGINE_CLOUD str
        +PRNU_NOISE str
        +MESONET str
    }

    class BaseDetectorPlugin {
        <<abstract>>
    }

    class LipSyncAnalyzer {
        +analyze_video(path) dict
    }

    class AudioDeepfakeAnalyzer {
        +analyze_audio(path) dict
    }

    class MetadataAnalyzer {
        +analyze(path) dict
    }

    class TemporalCoherenceAnalyzer {
        +analyze_frames(frames) dict
    }

    class rPPGAnalyzer {
        +analyze_face_track(rois, fps) dict
    }

    class FastAPIApp {
        <<entrypoint>>
        +POST /api/analyze
        +GET /api/result
        +GET /api/progress
        +GET /api/progress/stream
        +GET /api/frame
        +GET /health
    }

    PluginManager o-- BaseDetectorPlugin
    PluginManager *-- FacePreProcessor
    PluginManager ..> SceneClassifier
    SceneClassifier ..> SceneType
    SceneClassifier ..> PluginNames

    FastAPIApp *-- PluginManager
    FastAPIApp *-- LipSyncAnalyzer
    FastAPIApp *-- AudioDeepfakeAnalyzer
    FastAPIApp *-- MetadataAnalyzer
    FastAPIApp *-- TemporalCoherenceAnalyzer
    FastAPIApp *-- rPPGAnalyzer
```

---

## Padrões de design aplicados

| Padrão | Onde | Porquê |
|--------|------|--------|
| **Plugin / Strategy** | `BaseDetectorPlugin` + 6 implementações | Permite adicionar detetores sem alterar o `PluginManager`. Drop-in: basta colocar `.py` em `engine/plugins/`. |
| **Auto-discovery** | `PluginManager._load_all_plugins` | Reflexão em tempo de arranque: carrega todas as subclasses de `BaseDetectorPlugin` encontradas em `plugins/`. |
| **Template Method** | `BaseDetectorPlugin.analyze_frames_batch` | Default loopa `analyze_frame`; plugins com inference batched (MesoNet, ViT) sobrepõem. |
| **Façade** | `FacePreProcessor` | Esconde a complexidade MTCNN ↔ SSD fallback atrás de um único `process()`. |
| **Singleton (implícito)** | Instâncias em [main.py](../../engine/main.py) (linhas 45-50) | Plugins e analyzers instanciados uma única vez no arranque do FastAPI. |
| **Constants Class** | `PluginNames` | Evita strings mágicas em `SCENE_PLUGIN_WEIGHTS`; rename de plugin é detectável estaticamente. |
| **Composite Score** | `_run_analysis_locked` (weighted sum) | Combina scores heterogéneos numa pontuação final por face/frame. |

## Notas de design

### Porquê dois diagramas em vez de um
O diagrama unificado tinha ~30 classes e ~20 relações; o render no VS Code (Mermaid) ficava instável. Dividir em **hierarquia** + **orquestração** mantém ambos legíveis e garantidos a renderizar — e mapeia naturalmente às duas camadas arquiteturais (extensão vs coordenação).

### Composição interna do MesoNet (`_Meso4`)
O plugin é a fachada estável; a arquitetura PyTorch `_Meso4` (4 blocos Conv→ReLU→BN→MaxPool + 2 Dense) é detalhe de implementação privado. Omitida do diagrama para reduzir ruído — ver [engine/plugins/mesonet_detector.py](../../engine/plugins/mesonet_detector.py) linhas 85-133 para definição completa.

### Porque é que pesos não estão na classe do plugin
`plugin_weight()` existe na interface mas o cálculo real do score combinado usa a tabela `SCENE_PLUGIN_WEIGHTS` em [scene_classifier.py](../../engine/core/scene_classifier.py). Razão: os pesos dependem do **cenário**, não apenas do plugin. Um plugin pode pesar 0.40 em `CROPPED_FACE` e 0.0 em `NO_FACE`. A propriedade `plugin_weight()` é metadata informativa.

### Porque é que `SceneType` é `str` Enum
Para serializar diretamente em JSON nas respostas da API sem conversão manual. `scene_detected` aparece como `"CROPPED_FACE"` no payload retornado ao frontend.

### Endpoints REST omitem path parameters
As rotas estão listadas sem `:task_id` / `:analysis_id` por simplicidade visual. Rotas completas:
- `GET /api/result/{task_id}`
- `GET /api/frame/{analysis_id}/{frame_index}`
