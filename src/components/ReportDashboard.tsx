"use client";

import React from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { AlertCircle, CheckCircle2, ShieldAlert, AlertTriangle, Cpu, Activity } from 'lucide-react';
import VideoForensicsPlayer from './VideoForensicsPlayer';
import type { AnalysisReport, FrameDetail, PluginResult } from '@/types/report';
import { classifyScore, verdictTextClass, verdictBgClass } from '@/utils/verdict';

interface ReportProps {
  report: AnalysisReport;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function ReportDashboard({ report }: ReportProps) {
  const { results, metadata, lip_sync, audio_deepfake, metadata_forensics, temporal_coherence, rppg } = report;

  // Fallback to 0 if there was an error and overall_score is missing
  const overallScore = typeof results.overall_score === 'number' ? results.overall_score : 0;
  const overallVerdict = classifyScore(overallScore);
  const isFake = overallVerdict.isFake;

  // Calculate display values as numbers; format only at render time.
  const visualScoreNum = typeof results.visual_score === 'number' ? results.visual_score : overallScore;
  const visualPercentage = (visualScoreNum * 100).toFixed(1);
  const visualVerdict = classifyScore(visualScoreNum);

  // Audio percentage: hide when ALL audio modalities are inconclusive or missing.
  // Both lip_sync and audio_deepfake (WavLM) need to declare themselves
  // inconclusive for us to hide the score — if either has real signal, we show it.
  const lipSyncConclusive = !!lip_sync && !lip_sync.inconclusive;
  const wavlmConclusive = !!audio_deepfake && !audio_deepfake.inconclusive;
  const audioScoreVisible =
    typeof results.audio_score === 'number' && (lipSyncConclusive || wavlmConclusive);
  const audioScoreNum = audioScoreVisible ? (results.audio_score as number) : null;
  const audioPercentage = audioScoreNum !== null ? (audioScoreNum * 100).toFixed(1) : null;
  const audioVerdict = audioScoreNum !== null ? classifyScore(audioScoreNum) : null;

  // Plugins are pre-filtered by the backend
  const activePlugins: PluginResult[] = results.plugins ?? [];
  const frameDetails: FrameDetail[] = results.frame_details ?? [];

  // Build chart data — frame_details.overall_score is the max face score per frame
  const chartData = frameDetails.map((fd, i) => ({
    frame: `F${i + 1}`,
    score: parseFloat((fd.overall_score * 100).toFixed(1)),
  }));

  const maxFacesPerFrame = frameDetails.length > 0
    ? frameDetails.reduce((max, f) => Math.max(max, f.faces?.length ?? 0), 0)
    : 0;

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="w-full max-w-5xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">

      {/* Header Verdict Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">

        <div
          className={`p-6 rounded-2xl glass border-l-4 ${verdictBgClass(overallVerdict)} flex flex-col justify-center items-center text-center shadow-lg`}
          role="status"
          aria-live="polite"
          aria-label={`Veredito forense: ${overallVerdict.description}, score ${(overallScore * 100).toFixed(1)} porcento`}
        >
          {overallVerdict.level === 'strong' && <ShieldAlert className="w-12 h-12 text-red-500 mb-3" aria-hidden="true" />}
          {overallVerdict.level === 'suspect' && <AlertTriangle className="w-12 h-12 text-orange-500 mb-3" aria-hidden="true" />}
          {overallVerdict.level === 'inconclusive' && <AlertCircle className="w-12 h-12 text-yellow-500 mb-3" aria-hidden="true" />}
          {overallVerdict.level === 'clean' && <CheckCircle2 className="w-12 h-12 text-green-500 mb-3" aria-hidden="true" />}
          <h3 className="text-zinc-400 text-sm font-bold uppercase tracking-widest mb-1">Veredito Forense</h3>
          <p className={`text-xl font-black tracking-tight ${verdictTextClass(overallVerdict)}`}>
            {overallVerdict.label}
          </p>
          <p className="text-zinc-400 text-xs mt-2 max-w-[14rem]">{overallVerdict.description}</p>
          {results.triggered_by && overallVerdict.isFake && (
            <p className="text-[10px] text-zinc-500 uppercase tracking-wider mt-3">
              Sinal dominante:{' '}
              <span className={`font-bold ${verdictTextClass(overallVerdict)}`}>
                {results.triggered_by === 'visual' && 'Visual (Face/Píxeis)'}
                {results.triggered_by === 'wavlm' && 'Áudio (Clonagem Voz)'}
                {results.triggered_by === 'lip_sync' && 'Áudio (Sincronia Labial)'}
              </span>
            </p>
          )}
        </div>

        <div
          className="p-6 rounded-2xl glass flex flex-col justify-center items-center text-center"
          role="group"
          aria-label={`Risco visual ${visualPercentage} porcento, ${visualVerdict.description}`}
        >
          <h3 className="text-zinc-400 text-sm font-medium uppercase tracking-wider">Risco Visual</h3>
          <p className={`text-5xl font-black mt-1 ${verdictTextClass(visualVerdict)}`} aria-hidden="true">{visualPercentage}%</p>
          <p className={`text-[10px] font-bold uppercase tracking-widest mt-1 ${verdictTextClass(visualVerdict)}`} aria-hidden="true">
            {visualVerdict.label}
          </p>
          <p className="text-zinc-500 text-xs mt-2">
            Face e Píxeis
            {maxFacesPerFrame > 0 && (
              <span className="ml-2 font-medium text-purple-400">
                · {maxFacesPerFrame} pessoa(s)
              </span>
            )}
          </p>
        </div>

        <div
          className="p-6 rounded-2xl glass flex flex-col justify-center items-center text-center"
          role="group"
          aria-label={audioPercentage !== null && audioVerdict
            ? `Risco de áudio ${audioPercentage} porcento, ${audioVerdict.description}`
            : 'Sem áudio para analisar'}
        >
          <h3 className="text-zinc-400 text-sm font-medium uppercase tracking-wider">Risco de Áudio</h3>
          {audioPercentage !== null && audioVerdict ? (
            <>
              <p className={`text-5xl font-black mt-1 ${verdictTextClass(audioVerdict)}`} aria-hidden="true">{audioPercentage}%</p>
              <p className={`text-[10px] font-bold uppercase tracking-widest mt-1 ${verdictTextClass(audioVerdict)}`} aria-hidden="true">
                {audioVerdict.label}
              </p>
            </>
          ) : (
            <p className="text-3xl font-bold text-zinc-600 mt-1 uppercase">Sem Áudio</p>
          )}
          <p className="text-zinc-500 text-xs mt-2">Voz e Sincronia Labial</p>
        </div>

        <div className="p-6 rounded-2xl glass flex flex-col justify-center items-center text-center">
          <AlertCircle className="w-8 h-8 text-yellow-500 mb-2" />
          <h3 className="text-zinc-400 text-sm font-medium uppercase tracking-wider">Frames Analisados</h3>
          <p className="text-3xl font-bold text-white mt-1">{metadata.frames_analyzed}</p>
          {metadata.total_frames && metadata.total_frames !== metadata.frames_analyzed ? (
            <p className="text-zinc-500 text-xs mt-2">de {metadata.total_frames} totais · {metadata.duration_seconds?.toFixed(1)}s</p>
          ) : (
            <p className="text-zinc-500 text-xs mt-2">{metadata.filename} · {formatBytes(metadata.filesize_bytes)}</p>
          )}
          {metadata.resolution && (
            <p className="text-zinc-600 text-xs">{metadata.resolution} · {metadata.fps?.toFixed(0)} fps</p>
          )}
        </div>

      </div>

      {/* Video Forensics Player */}
      {frameDetails.length > 0 && (
        <div className="p-6 rounded-2xl glass w-full">
          <VideoForensicsPlayer
            analysisId={report.analysis_id}
            frameDetails={frameDetails}
            pluginNames={activePlugins.map((p) => p.name)}
          />
        </div>
      )}

      {/* Frame Score Chart */}
      {chartData.length > 0 && (
        <div className="p-6 rounded-2xl glass w-full">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-purple-400" />
              Distribuição de Suspeita por Frame
            </h2>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor={isFake ? "#ef4444" : "#10b981"} stopOpacity={0.8} />
                    <stop offset="95%" stopColor={isFake ? "#ef4444" : "#10b981"} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                <XAxis dataKey="frame" stroke="#71717a" fontSize={12} tickMargin={10} />
                <YAxis stroke="#71717a" fontSize={12} domain={[0, 100]} unit="%" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px', color: '#fff' }}
                  itemStyle={{ color: isFake ? '#ef4444' : '#10b981' }}
                  formatter={(value: unknown) => [`${Number(value).toFixed(1)}%`, "Prob. Falso"]}
                />
                <Area
                  type="monotone"
                  dataKey="score"
                  name="Probabilidade (%)"
                  stroke={isFake ? "#ef4444" : "#10b981"}
                  strokeWidth={3}
                  fillOpacity={1}
                  fill="url(#colorScore)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Plugin Results Breakdown */}
      <div className="p-6 rounded-2xl glass w-full">
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <Cpu className="w-5 h-5 text-purple-400" />
          Resumo Global dos Plugins
        </h2>
        <div className="space-y-4">
          {activePlugins.map((plugin, i) => {
            const rawScore = plugin.average_score;
            const score = typeof rawScore === 'number' && !isNaN(rawScore) ? rawScore : 0.5;
            const pluginPct = (score * 100).toFixed(1);
            const pluginVerdict = classifyScore(score);
            const barColor = pluginVerdict.colorKey === 'green' ? 'bg-green-500'
                          : pluginVerdict.colorKey === 'yellow' ? 'bg-yellow-500'
                          : pluginVerdict.colorKey === 'orange' ? 'bg-orange-500'
                          : 'bg-red-500';
            return (
              <div key={i} className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800 hover:border-zinc-700 transition-colors">
                <div className="flex justify-between items-start flex-wrap gap-2">
                  <div>
                    <h3 className="font-bold text-white text-sm">{plugin.name}</h3>
                    <p className="text-zinc-500 text-xs mt-1 max-w-xl">
                      Média calculada sobre {plugin.frames_analyzed} deteções de caras.
                    </p>
                  </div>
                  <span className={`text-xl font-black shrink-0 ${verdictTextClass(pluginVerdict)}`}>
                    {pluginPct}%
                  </span>
                </div>
                <div className="mt-3 h-2 rounded-full bg-zinc-800 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-1000 ${barColor}`}
                    style={{ width: `${pluginPct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Additional Forensic Signals */}
      {(metadata_forensics || temporal_coherence || rppg) && (
        <div className="p-6 rounded-2xl glass w-full mt-6">
          <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-purple-400" aria-hidden="true" />
            Sinais Forenses Adicionais
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

            {metadata_forensics && (() => {
              const v = classifyScore(metadata_forensics.metadata_score);
              return (
                <div
                  className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800"
                  role="group"
                  aria-label={`Metadata: ${v.description}, score ${(metadata_forensics.metadata_score * 100).toFixed(1)} porcento`}
                >
                  <h3 className="font-bold text-white text-sm">Metadata (EXIF/Codec)</h3>
                  <p className="text-zinc-500 text-xs mt-1">
                    {metadata_forensics.signals.length > 0
                      ? metadata_forensics.signals.slice(0, 3).join(', ')
                      : 'Sem sinais suspeitos'}
                  </p>
                  <p className={`text-3xl font-black mt-2 ${verdictTextClass(v)}`} aria-hidden="true">
                    {(metadata_forensics.metadata_score * 100).toFixed(1)}%
                  </p>
                </div>
              );
            })()}

            {temporal_coherence && (() => {
              const v = classifyScore(temporal_coherence.temporal_score);
              return (
                <div
                  className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800"
                  role="group"
                  aria-label={`Coerência temporal: ${v.description}, score ${(temporal_coherence.temporal_score * 100).toFixed(1)} porcento`}
                >
                  <h3 className="font-bold text-white text-sm">Coerência Temporal</h3>
                  <p className="text-zinc-500 text-xs mt-1">
                    {temporal_coherence.signals.length > 0
                      ? temporal_coherence.signals.slice(0, 2).join(', ')
                      : 'Frames consistentes entre si'}
                  </p>
                  <p className={`text-3xl font-black mt-2 ${verdictTextClass(v)}`} aria-hidden="true">
                    {(temporal_coherence.temporal_score * 100).toFixed(1)}%
                  </p>
                </div>
              );
            })()}

            {rppg && (() => {
              const v = classifyScore(rppg.rppg_score);
              return (
                <div
                  className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800"
                  role="group"
                  aria-label={`rPPG sinal fisiológico: ${v.description}, score ${(rppg.rppg_score * 100).toFixed(1)} porcento`}
                >
                  <h3 className="font-bold text-white text-sm">Pulso Fisiológico (rPPG)</h3>
                  <p className="text-zinc-500 text-xs mt-1">
                    {rppg.estimated_bpm
                      ? `BPM estimado: ${rppg.estimated_bpm}`
                      : (rppg.reason || 'Sinal insuficiente')}
                  </p>
                  <p className={`text-3xl font-black mt-2 ${verdictTextClass(v)}`} aria-hidden="true">
                    {(rppg.rppg_score * 100).toFixed(1)}%
                  </p>
                </div>
              );
            })()}

          </div>
        </div>
      )}

      {/* Audio Visual Analysis */}
      {(lip_sync || audio_deepfake) && (
        <div className="p-6 rounded-2xl glass w-full mt-6">
          <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-purple-400" />
            Análise Áudio-Visual
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            {audio_deepfake && (() => {
              const isInconclusive = audio_deepfake.inconclusive === true;
              const v = classifyScore(audio_deepfake.audio_fake_score);
              const barColor = v.colorKey === 'green' ? 'bg-green-500'
                            : v.colorKey === 'yellow' ? 'bg-yellow-500'
                            : v.colorKey === 'orange' ? 'bg-orange-500'
                            : 'bg-red-500';
              // Map common reason codes to user-readable text.
              const reasonText: Record<string, string> = {
                no_speech_detected: 'Sem fala humana detetada no áudio (silêncio / ruído / música).',
              };
              return (
              <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800 hover:border-zinc-700 transition-colors">
                <div className="flex justify-between items-start flex-wrap gap-2">
                  <div>
                    <h3 className="font-bold text-white text-sm">Clonagem de Voz (WavLM)</h3>
                    <p className="text-zinc-500 text-xs mt-1 max-w-xs">
                      {isInconclusive
                        ? (audio_deepfake.reason && reasonText[audio_deepfake.reason]) ||
                          'Áudio não classificável (não contém fala humana).'
                        : 'Deteta vozes geradas por IA (VALL-E, ElevenLabs, etc.)'}
                    </p>
                  </div>
                  {isInconclusive ? (
                    <span className="text-lg font-bold text-yellow-500 shrink-0 uppercase">Inconclusivo</span>
                  ) : (
                    <span className={`text-xl font-black shrink-0 ${verdictTextClass(v)}`}>
                      {(audio_deepfake.audio_fake_score * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
                {!isInconclusive && (
                  <div className="mt-3 h-2 rounded-full bg-zinc-800 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-1000 ${barColor}`}
                      style={{ width: `${(audio_deepfake.audio_fake_score * 100).toFixed(1)}%` }}
                    />
                  </div>
                )}
              </div>
              );
            })()}

            {lip_sync && (() => {
              const v = !lip_sync.inconclusive ? classifyScore(lip_sync.lip_sync_score) : null;
              const barColor = v?.colorKey === 'green' ? 'bg-green-500'
                            : v?.colorKey === 'yellow' ? 'bg-yellow-500'
                            : v?.colorKey === 'orange' ? 'bg-orange-500'
                            : 'bg-red-500';
              return (
              <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800 hover:border-zinc-700 transition-colors">
                <div className="flex justify-between items-start flex-wrap gap-2">
                  <div>
                    <h3 className="font-bold text-white text-sm">Sincronia Labial (Lip Sync)</h3>
                    <p className="text-zinc-500 text-xs mt-1 max-w-xs">
                      {lip_sync.inconclusive
                        ? (lip_sync.reason === 'no_speech_detected'
                            ? 'Sem fala humana detetada no áudio — não há sinal para correlacionar.'
                            : 'Dados insuficientes para medir correlação (poucas bocas detetadas ou áudio constante).')
                        : `Mede a correlação (${lip_sync.correlation.toFixed(2)}) entre a voz e o movimento dos lábios.`}
                    </p>
                  </div>
                  {lip_sync.inconclusive || !v ? (
                    <span className="text-lg font-bold text-yellow-500 shrink-0 uppercase">Inconclusivo</span>
                  ) : (
                    <span className={`text-xl font-black shrink-0 ${verdictTextClass(v)}`}>
                      {(lip_sync.lip_sync_score * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
                {!lip_sync.inconclusive && (
                  <div className="mt-3 h-2 rounded-full bg-zinc-800 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-1000 ${barColor}`}
                      style={{ width: `${(lip_sync.lip_sync_score * 100).toFixed(1)}%` }}
                    />
                  </div>
                )}
              </div>
              );
            })()}

          </div>
        </div>
      )}

    </div>
  );
}
