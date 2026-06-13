# Diagrama de Classes — Arquitetura do Engine

Representa a estrutura orientada a objetos do backend (Cap. 3.4 do relatório).
Por legibilidade e fiabilidade de render, está dividido em dois diagramas:

1. **Hierarquia de plugins** — contrato `BaseDetectorPlugin` e as 6 implementações concretas.
2. **Orquestração + analyzers vídeo-level** — `PluginManager`, `FacePreProcessor`, `SceneClassifier`, e os analyzers que correm sobre o vídeo completo.

---

## 1. Hierarquia de plugins

```mermaid
%%{init: {'themeVariables': {'fontSize': '22px', 'fontFamily': 'Inter, Arial, sans-serif'}}}%%
classDiagram
    class BaseDetectorPlugin {
        <<abstract>>
        +SUPPORTS_BATCH
        +plugin_name()
        +plugin_description()
        +plugin_version()
        +plugin_weight()
        +analyze_frame()
        +analyze_frames_batch()
        +is_configured()
        +reset()
        +get_plugin_info()
    }

    class MesoNetDetectorPlugin {
        +analyze_frame()
        +analyze_frames_batch()
        -_preprocess()
    }

    class ViTDetectorPlugin {
        +analyze_frame()
        +analyze_frames_batch()
        -_calibrate()
    }

    class DCTFrequencyAnalyzerPlugin {
        +analyze_frame()
        -_power_law_deviation()
        -_stride_artifact_score()
        -_spectral_flatness_score()
    }

    class EdgeBlendingDetector {
        +analyze_frame()
        -_gradient_discontinuity_score()
        -_color_boundary_score()
        -_illumination_consistency_score()
    }

    class PRNUNoiseResidueDetector {
        +analyze_frame()
        -_noise_ratio_score()
        -_locate_roi()
    }

    class SightengineDeepfakeDetector {
        +analyze_frame()
        +reset()
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
%%{init: {'themeVariables': {'fontSize': '22px', 'fontFamily': 'Inter, Arial, sans-serif'}}}%%
classDiagram
    class PluginManager {
        +run_analysis()
        +get_plugins()
        -_load_all_plugins()
        -_run_analysis_locked()
    }

    class FacePreProcessor {
        +process()
        -_process_mtcnn()
        -_process_ssd()
    }

    class SceneClassifier {
        +classify()
        +get_active_plugins_and_weights()
    }

    class SceneType {
        <<enumeration>>
        CROPPED_FACE
        FACE_IN_SCENE
        NO_FACE
    }

    class PluginNames {
        <<constants>>
        VIT_DETECTOR
        DCT_FREQUENCY
        EDGE_BLENDING
        SIGHTENGINE_CLOUD
        PRNU_NOISE
        MESONET
    }

    class BaseDetectorPlugin {
        <<abstract>>
    }

    PluginManager o-- BaseDetectorPlugin
    PluginManager *-- FacePreProcessor
    PluginManager ..> SceneClassifier
    SceneClassifier ..> SceneType
    SceneClassifier ..> PluginNames
```

---

## 3. API REST e analyzers vídeo-level

```mermaid
%%{init: {'themeVariables': {'fontSize': '22px', 'fontFamily': 'Inter, Arial, sans-serif'}}}%%
classDiagram
    class FastAPIApp {
        <<entrypoint>>
        +POST /api/analyze
        +GET /api/result
        +GET /api/progress
        +GET /api/progress/stream
        +GET /api/frame
        +GET /health
    }

    class PluginManager {
        +run_analysis()
    }

    class LipSyncAnalyzer {
        +analyze_video()
    }

    class AudioDeepfakeAnalyzer {
        +analyze_audio()
    }

    class MetadataAnalyzer {
        +analyze()
    }

    class TemporalCoherenceAnalyzer {
        +analyze_frames()
    }

    class rPPGAnalyzer {
        +analyze_face_track()
    }

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
