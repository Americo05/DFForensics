// Shared analysis-report types used by the analyzer page, the dashboard,
// and the PDF report generator. Keep in sync with engine/main.py response shape.

export interface FaceBBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FaceDetail {
  face_bbox: FaceBBox | null;
  scene_detected: string;
  overall_score: number;
  plugin_scores: Record<string, number | string>;
}

export interface FrameDetail {
  frame_index: number;
  timestamp_seconds: number;
  faces: FaceDetail[];
  overall_score: number;
}

export interface PluginResult {
  name: string;
  average_score: number;
  frames_analyzed: number;
}

export interface AnalysisMetadata {
  filename: string;
  filesize_bytes: number;
  format: string;
  frames_analyzed: number;
  total_frames?: number;
  fps?: number;
  resolution?: string;
  duration_seconds?: number;
  extraction_method?: string;
  analysis_mode?: string;
  active_plugins: number;
}

export type TriggeredBy = 'visual' | 'wavlm' | 'lip_sync';

export interface AnalysisResults {
  overall_score: number;
  visual_score?: number;
  audio_score?: number | null;
  /** Which modality "won" the MAX aggregation (drove the verdict). */
  triggered_by?: TriggeredBy;
  /** The score from that modality (for display alongside the label). */
  triggered_by_score?: number;
  plugins: PluginResult[];
  frame_details?: FrameDetail[];
}

export interface LipSyncResult {
  lip_sync_score: number;
  correlation: number;
  frames_analyzed: number;
  inconclusive?: boolean;
  reason?: string;
  windows_evaluated?: number;
  verdict: string;
}

export interface AudioDeepfakeResult {
  audio_fake_score: number;
  verdict: string;
  inconclusive?: boolean;
  reason?: string;
  chunks_evaluated?: number;
  worst_chunk_start_s?: number;
  worst_chunk_end_s?: number;
}

export interface MetadataForensicsResult {
  metadata_score: number;
  signals: string[];
  details?: Record<string, unknown>;
  verdict: string;
}

export interface TemporalCoherenceResult {
  temporal_score: number;
  signals: string[];
  frame_count: number;
  components?: Record<string, number>;
  verdict: string;
}

export interface RppgResult {
  rppg_score: number;
  estimated_bpm?: number;
  peak_prominence?: number;
  verdict: string;
  frames_analyzed?: number;
  reason?: string;
}

export interface AnalysisReport {
  analysis_id: string;
  status: string;
  error?: string;
  metadata: AnalysisMetadata;
  results: AnalysisResults;
  lip_sync?: LipSyncResult | null;
  audio_deepfake?: AudioDeepfakeResult | null;
  metadata_forensics?: MetadataForensicsResult | null;
  temporal_coherence?: TemporalCoherenceResult | null;
  rppg?: RppgResult | null;
}
