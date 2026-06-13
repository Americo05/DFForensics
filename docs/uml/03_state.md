# Diagrama de Estados — Ciclo de vida de uma análise

Descreve as transições possíveis de uma análise (`task_id`) desde a sua criação até à expiração (Cap. 3.3 do relatório).

Os estados correspondem ao campo `stage` propagado via SSE em `/api/progress/stream` (ver [main.py](../../engine/main.py)).

```mermaid
%%{init: {'themeVariables': {'fontSize': '26px', 'fontFamily': 'Inter, Arial, sans-serif'}, 'stateDiagram': {'padding': 20}}}%%
stateDiagram-v2
    [*] --> IDLE: Cliente acede ao Dashboard

    IDLE --> UPLOADING: POST /api/analyze
    UPLOADING --> ERROR: Auth/RateLimit/Size falham
    UPLOADING --> EXTRACTING: Ficheiro persistido<br/>task_id criado

    EXTRACTING --> ERROR: cv2.VideoCapture falha
    EXTRACTING --> ANALYZING: Frames extraídos

    ANALYZING --> ANALYZING: Próximo frame<br/>(progress_callback)
    ANALYZING --> AUDIO: Loop de frames completo<br/>(se vídeo)
    ANALYZING --> COMPLETED: (se imagem,<br/>sem analyzers de áudio)
    ANALYZING --> ERROR: Plugin crashou irrecuperável

    AUDIO --> COMPLETED: LipSync + AudioDeepfake terminam
    AUDIO --> ERROR: ffprobe ausente / áudio corrompido

    COMPLETED --> EXPIRED: TTL atingido (5 min)
    ERROR --> EXPIRED: TTL atingido (5 min)
    EXPIRED --> [*]: Resultado removido<br/>de memória

    note right of UPLOADING
        Validações:
        • X-API-Key (se ENGINE_API_KEYS definido)
        • Sliding-window rate limit
        • Content-Length ≤ MAX_UPLOAD_BYTES
        • Streaming size cap
    end note

    note right of ANALYZING
        Por cada frame:
        1. FacePreProcessor (MTCNN)
        2. SceneClassifier
        3. Plugins ativos
           (batched quando aplicável)
        4. Agrega MAX por frame
        5. progress_callback → SSE
    end note

    note left of AUDIO
        Só executa se has_audio_track:
        • LipSyncAnalyzer (MediaPipe)
        • AudioDeepfakeAnalyzer (Silero VAD)
    end note

    note right of COMPLETED
        Resultado persistido em
        OrderedDict (memória).
        Disponível em:
        GET /api/result/:task_id
    end note
```

## Tabela de transições

| Estado origem | Evento | Estado destino | Ação |
|----------------|--------|----------------|------|
| `IDLE` | `POST /api/analyze` | `UPLOADING` | Validar request, ler stream |
| `UPLOADING` | Auth falha | `ERROR` | HTTP 401 |
| `UPLOADING` | Rate limit excedido | `ERROR` | HTTP 429 |
| `UPLOADING` | Tamanho excede limite | `ERROR` | HTTP 413 |
| `UPLOADING` | Stream lido OK | `EXTRACTING` | Cria task_id, agenda background task |
| `EXTRACTING` | `cv2.VideoCapture` falha | `ERROR` | Devolve mensagem ao cliente via SSE |
| `EXTRACTING` | N frames extraídos | `ANALYZING` | Inicia loop de plugins |
| `ANALYZING` | Frame processado | `ANALYZING` | Emite progresso (X/N) |
| `ANALYZING` | Loop completo (vídeo) | `AUDIO` | Verifica se faixa de áudio existe |
| `ANALYZING` | Loop completo (imagem) | `COMPLETED` | Skip analyzers de áudio |
| `ANALYZING` | Exceção em plugin | `ANALYZING` | Score neutral 0.5 (continua); plugin marcado em `plugin_errors` |
| `AUDIO` | Analyzers OK | `COMPLETED` | Persiste payload final |
| `AUDIO` | ffprobe ausente | `COMPLETED` | Análise de áudio omitida; vídeo continua válido |
| `COMPLETED` | Cliente fez `GET /api/result` | `COMPLETED` | Devolve payload (mas marca para limpeza) |
| `COMPLETED` / `ERROR` | TTL 5 min | `EXPIRED` | Remove de memória, libera /tmp |
| `EXPIRED` | — | `[*]` | Fim do ciclo |

## Pontos de robustez documentados

### Falha graciosa de plugin
Quando um plugin lança exceção em `analyze_frame`, a análise **não aborta**. O `plugin_manager._run_analysis_locked` ([linha 451-464 plugin_manager.py](../../engine/core/plugin_manager.py#L451)):
1. Captura exceção
2. Incrementa `plugin_errors[plugin.plugin_name]`
3. Loga apenas a primeira falha por plugin (evita flood)
4. Usa score `0.5` neutro como fallback
5. Continua para o próximo plugin/frame

### Idempotência de `reset()`
Antes de cada análise, todos os plugins recebem `reset()` ([plugin_manager.py linha 352-356](../../engine/core/plugin_manager.py#L352)). Plugins stateful (ex: Sightengine, que mantém `_frame_counter` e `_cached_score`) limpam estado para evitar leakage entre análises sucessivas.

### TTL e limpeza de memória
Resultados completos vivem 5 minutos em `OrderedDict` LRU. Ficheiros temporários em `/tmp/<uuid>` são removidos imediatamente após análise terminar (sucesso ou erro).
