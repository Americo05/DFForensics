"use client";

import React, { useState, useEffect, useRef } from 'react';
import ReportDashboard from '@/components/ReportDashboard';
import TopNav from '@/components/TopNav';
import DemoModeBanner from '@/components/DemoModeBanner';
import { UploadCloud, FileVideo, Loader2, Cloud, CloudOff, FileDown } from 'lucide-react';
import type { AnalysisReport } from '@/types/report';
import { isDemoMode } from '@/utils/demoMode';

// ── Progress Types ───────────────────────────────────────────────────────────

interface ProgressData {
  active: boolean;
  stage: string;
  current_frame: number;
  total_frames: number;
  message: string;
}

import { useTheme } from '@/hooks/useTheme';

// ── API Base ─────────────────────────────────────────────────────────────────

function getApiBase(): string {
  if (typeof window === "undefined") return "http://localhost:8000";
  const host = window.location.hostname === "localhost" ? "127.0.0.1" : window.location.hostname;
  return `http://${host}:8000`;
}

// Optional API key. When the backend has ENGINE_API_KEYS set, every request
// needs to include X-API-Key. The value is read from NEXT_PUBLIC_ENGINE_API_KEY
// at build time; leaving it unset means "no auth", which matches the backend's
// default. Public-facing deployments MUST set both.
const API_KEY: string | undefined = process.env.NEXT_PUBLIC_ENGINE_API_KEY;

function apiHeaders(extra: HeadersInit = {}): HeadersInit {
  if (!API_KEY) return extra;
  return { ...extra, 'X-API-Key': API_KEY };
}

const POLL_INTERVAL_MS = 400;
const POLL_MAX_CONSECUTIVE_ERRORS = 10; // ~4s of failed polls before surfacing

// ── Component ────────────────────────────────────────────────────────────────

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useCloudApi, setUseCloudApi] = useState(true);
  const [progress, setProgress] = useState<ProgressData | null>(null);
  // Demo mode = no local engine reachable (public Vercel deploy). Start as
  // true so SSR + hydration agree on the public site; flip to false in the
  // post-mount effect below when the page is actually served from localhost.
  const [demoMode, setDemoMode] = useState(true);
  useEffect(() => {
    setDemoMode(isDemoMode());
  }, []);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const resetTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Track which task the current poller belongs to, so a stale interval can't
  // overwrite progress for a newer analysis.
  const activeTaskIdRef = useRef<string | null>(null);
  // The poller calls into fetchResult on "done"/"error". Stored in a ref so
  // startPolling doesn't need to be re-created when fetchResult identity changes.
  const fetchResultRef = useRef<((taskId: string) => Promise<void>) | null>(null);
  const { isDark } = useTheme();

  // SSE handle — closed when a newer analysis starts or the task completes.
  const sseRef = useRef<EventSource | null>(null);

  // Prefer Server-Sent Events (one persistent connection). Fall back to
  // interval polling if EventSource is unavailable (very old browsers) OR
  // the SSE connection errors out before any message arrives — that's
  // typically a misconfigured proxy or auth-header issue (EventSource
  // can't send custom headers, so SSE only works when ENGINE_API_KEYS is
  // empty; auth-enabled deployments transparently degrade to polling).
  const startPolling = (taskId: string) => {
    const apiBase = getApiBase();
    activeTaskIdRef.current = taskId;
    let resultTriggered = false;

    const triggerResult = async () => {
      if (resultTriggered) return;
      resultTriggered = true;
      if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      const fr = fetchResultRef.current;
      if (fr) await fr(taskId);
    };

    const startFallbackPolling = () => {
      if (pollRef.current || resultTriggered) return;
      let consecutiveErrors = 0;
      pollRef.current = setInterval(async () => {
        if (activeTaskIdRef.current !== taskId) return;
        try {
          const res = await fetch(
            `${apiBase}/api/progress?task_id=${encodeURIComponent(taskId)}`,
            { headers: apiHeaders() },
          );
          if (!res.ok) throw new Error(`progress ${res.status}`);
          const data = await res.json() as ProgressData;
          if (activeTaskIdRef.current === taskId) {
            setProgress(data);
            consecutiveErrors = 0;
            const stage = (data as ProgressData & { stage?: string }).stage;
            if (stage === 'done' || stage === 'error') await triggerResult();
          }
        } catch (pollErr) {
          consecutiveErrors += 1;
          if (consecutiveErrors >= POLL_MAX_CONSECUTIVE_ERRORS) {
            console.warn('Progress polling failing repeatedly:', pollErr);
            if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
            if (activeTaskIdRef.current === taskId) {
              setError("Sem resposta do motor — verifica que o servidor está a correr.");
              activeTaskIdRef.current = null;
              setIsProcessing(false);
            }
          }
        }
      }, POLL_INTERVAL_MS);
    };

    // EventSource doesn't support custom headers, so it can only carry the
    // API key via the query string. If auth is enabled we'd be passing the
    // key in the URL — that's logged by proxies/access logs, undesirable.
    // Skip SSE entirely in that case and use polling (which uses headers).
    const canUseSSE = typeof window !== 'undefined'
      && typeof window.EventSource !== 'undefined'
      && !API_KEY;

    if (canUseSSE) {
      try {
        const es = new EventSource(`${apiBase}/api/progress/stream?task_id=${encodeURIComponent(taskId)}`);
        sseRef.current = es;
        let everReceivedMessage = false;
        // Tracks whether we observed THIS task reach done/error — distinguishes
        // "real completion" from "server-restart connection close".
        let sawTerminalStage = false;

        es.onmessage = (ev) => {
          if (activeTaskIdRef.current !== taskId) return;
          everReceivedMessage = true;
          try {
            const data = JSON.parse(ev.data) as ProgressData;
            setProgress(data);
            const stage = (data as ProgressData & { stage?: string }).stage;
            if (stage === 'done' || stage === 'error') {
              sawTerminalStage = true;
              void triggerResult();
            }
          } catch {
            // Ignore malformed events
          }
        };
        es.addEventListener('end', (ev) => {
          // Parse the reason; the backend distinguishes:
          //   "completed"     — task reached done/error, result is in the store
          //   "unknown_task"  — server doesn't know this task_id (e.g. restart)
          //   "deadline"      — stream timed out
          let reason: string | undefined;
          try {
            reason = JSON.parse((ev as MessageEvent).data || '{}').reason;
          } catch { /* ignore */ }

          if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }

          if (reason === 'completed' || sawTerminalStage) {
            void triggerResult();
          } else if (reason === 'unknown_task') {
            // Stale connection from a prior session — let the user retry.
            if (activeTaskIdRef.current === taskId) {
              setError('A análise não foi encontrada no servidor (o motor pode ter sido reiniciado). Tenta novamente.');
              activeTaskIdRef.current = null;
              setIsProcessing(false);
              stopPolling();
            }
          } else {
            // Deadline or unknown reason — fall back to polling, the result
            // might still arrive
            startFallbackPolling();
          }
        });
        es.onerror = () => {
          // Browser auto-reconnects EventSource. If we haven't received any
          // message yet, the stream is likely unsupported (e.g. CORS/proxy);
          // fall back to polling immediately.
          if (!everReceivedMessage && !resultTriggered) {
            if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }
            console.info('SSE failed before first message; falling back to polling');
            startFallbackPolling();
          }
        };
        return;
      } catch {
        // EventSource construction itself failed (very rare) — polling time.
      }
    }
    startFallbackPolling();
  };

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    if (sseRef.current) {
      sseRef.current.close();
      sseRef.current = null;
    }
    // Keep progress visible briefly for the "done" animation
    if (resetTimeoutRef.current) clearTimeout(resetTimeoutRef.current);
    resetTimeoutRef.current = setTimeout(() => setProgress(null), 1500);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (sseRef.current) sseRef.current.close();
      if (resetTimeoutRef.current) clearTimeout(resetTimeoutRef.current);
    };
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
      setError(null);
    }
  };

  const submitToEngine = async () => {
    if (!file || isProcessing) return;
    setIsProcessing(true);
    setReport(null);
    setError(null);

    // Generate a unique task ID for this analysis run
    const taskId = crypto.randomUUID();
    activeTaskIdRef.current = taskId;

    // Show an "uploading" placeholder so the user sees feedback during the
    // POST. The real progress takes over once the server starts pushing it.
    setProgress({
      active: true,
      stage: 'upload',
      current_frame: 0,
      total_frames: 0,
      message: 'A enviar ficheiro...',
    });

    try {
      const formData = new FormData();
      formData.append('file', file);

      const apiBase = getApiBase();
      // POST first and WAIT for the server to acknowledge. Only after the
      // 200 response do we open the SSE/polling channel — at that point the
      // server has already created the progress entry, so there's no race
      // where the stream asks for a task the server hasn't recorded yet.
      const response = await fetch(
        `${apiBase}/api/analyze?cloud=${useCloudApi}&task_id=${encodeURIComponent(taskId)}`,
        { method: 'POST', body: formData, headers: apiHeaders() },
      );

      if (!response.ok) {
        const errText = await response.text().catch(() => '');
        throw new Error(
          errText
            ? `O motor respondeu com ${response.status}: ${errText.slice(0, 200)}`
            : `O motor respondeu com o estado: ${response.status}`,
        );
      }
      // Discard the {queued, ...} body; the streamer drives state from here.
      await response.json().catch(() => null);

      // Server is ready — now start the stream.
      startPolling(taskId);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Erro de conexão ao Motor Analítico.";
      console.error("Falha na análise:", err);
      setError(message);
      activeTaskIdRef.current = null;
      setIsProcessing(false);
      stopPolling();
    }
  };

  // When the poller observes stage="done" or "error", fetch the final result.
  // Stored in a ref so the polling effect can call it without re-binding.
  //
  // Defense in depth: the backend now stores the result BEFORE flipping the
  // stage to "done", but if anything ever races (proxy buffering SSE events,
  // slow filesystem, etc.) we retry 404s a few times with backoff so the
  // user doesn't see a flash of "Resultado indisponível" before the real data
  // shows up.
  const fetchResult = async (taskId: string): Promise<void> => {
    const apiBase = getApiBase();
    const url = `${apiBase}/api/result/${encodeURIComponent(taskId)}`;
    const RETRY_DELAYS_MS = [150, 300, 600, 1200];

    try {
      let lastStatus = 0;
      for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt++) {
        const res = await fetch(url, { headers: apiHeaders() });
        if (res.ok) {
          const data = await res.json() as AnalysisReport;
          if (data.status === 'error') {
            throw new Error(data.error ?? 'Erro desconhecido no motor analítico.');
          }
          setReport(data);
          return;
        }
        lastStatus = res.status;
        // Only 404 is worth retrying — it means "not yet stored, try again".
        // Any other status is a real failure (401 auth, 5xx server) — abort.
        if (res.status !== 404 || attempt === RETRY_DELAYS_MS.length) break;
        await new Promise(r => setTimeout(r, RETRY_DELAYS_MS[attempt]));
      }
      throw new Error(`Resultado indisponível (${lastStatus})`);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Erro ao obter resultado.";
      console.error("Falha ao obter resultado:", err);
      setError(message);
    } finally {
      activeTaskIdRef.current = null;
      setIsProcessing(false);
      stopPolling();
    }
  };

  // Keep the latest fetchResult bound to the ref so the poller can call it.
  fetchResultRef.current = fetchResult;

  // Calculate progress percentage (guarded against division-by-zero / NaN)
  const progressPct = progress && progress.total_frames > 0
    ? Math.min(100, Math.max(0, Math.round((progress.current_frame / progress.total_frames) * 100)))
    : 0;

  return (
    <main className={`min-h-screen selection:bg-purple-500/30 transition-colors duration-300 ${
      isDark ? 'bg-[#050505] text-white' : 'bg-[#f8f9fb] text-zinc-900'
    }`}>

      {/* Dynamic Background Blurs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-purple-600/20 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-600/10 blur-[120px] rounded-full" />
      </div>

      {/* ── Top Navigation Bar ────────────────────────────────────────────── */}
      <TopNav />

      <div className="relative z-10 max-w-7xl mx-auto px-6 pt-28 pb-12 md:pt-32 md:pb-20 flex flex-col items-center">

        <div className="text-center max-w-3xl mb-12">
          <h1 className="text-5xl md:text-7xl font-black tracking-tight mb-6 bg-gradient-to-br from-white via-white to-zinc-500 bg-clip-text text-transparent"
            style={!isDark ? { backgroundImage: 'linear-gradient(to bottom right, #18181b, #18181b, #71717a)' } : undefined}
          >
            Deteção de Deepfakes
          </h1>
          <p className="text-lg text-zinc-400 font-light">
            {demoMode
              ? 'Versão pública informativa. Instala localmente para analisar conteúdo.'
              : 'Faça upload do seu ficheiro de vídeo ou imagem'}
          </p>
        </div>

        {/* Demo mode replaces the entire upload/analysis surface with a CTA
            pointing visitors at the local install — the engine isn't reachable
            from the public deploy and a "drag a file here" button that fails
            silently would be worse UX than no button at all. */}
        {demoMode && (
          <div className="w-full">
            <DemoModeBanner
              feature="Análise de vídeo e imagem"
              reason="O detetor combina 6 plugins ML (MesoNet, ViT, DCT, Edge Blending, PRNU, Sightengine) com 5 analyzers vídeo-level (lip-sync, áudio, rPPG, temporal, metadata). Tudo corre no teu computador para preservar privacidade do material analisado."
            />
          </div>
        )}

        {/* Upload Zone */}
        {!demoMode && !report && (
          <div className="w-full max-w-2xl mb-12">
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className={`
                relative group overflow-hidden rounded-3xl border-2 border-dashed transition-all duration-300
                ${file ? 'border-purple-500 bg-purple-500/5' : isDark ? 'border-zinc-800 hover:border-zinc-700 bg-zinc-900/30 hover:bg-zinc-900/50' : 'border-zinc-300 hover:border-zinc-400 bg-white/40 hover:bg-white/60'}
                glass p-12 flex flex-col items-center justify-center text-center cursor-pointer
              `}
              onClick={() => document.getElementById('fileUpload')?.click()}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') document.getElementById('fileUpload')?.click(); }}
              role="button"
              tabIndex={0}
              aria-label={file ? `Ficheiro selecionado: ${file.name}` : 'Selecionar ou arrastar ficheiro de vídeo/imagem para análise'}
            >
              <input
                id="fileUpload"
                type="file"
                accept="video/*,image/*"
                className="hidden"
                onChange={(e) => { if (e.target.files) { setFile(e.target.files[0]); setError(null); } }}
              />

              {file ? (
                <>
                  <div className="w-16 h-16 rounded-full bg-purple-500/20 flex items-center justify-center mb-4">
                    <FileVideo className="w-8 h-8 text-purple-400" />
                  </div>
                  <h3 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-zinc-900'}`}>{file.name}</h3>
                  <p className="text-zinc-400 text-sm mt-2">Ficheiro pronto para análise.</p>
                </>
              ) : (
                <>
                  <div className={`w-16 h-16 rounded-full glass border flex items-center justify-center mb-4 group-hover:scale-110 transition-transform ${isDark ? 'border-zinc-800' : 'border-zinc-300'}`}>
                    <UploadCloud className="w-8 h-8 text-zinc-400 group-hover:text-purple-400 transition-colors" />
                  </div>
                  <h3 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-zinc-900'}`}>Arraste um vídeo ou imagem</h3>
                  <p className="text-zinc-400 text-sm mt-2">ou clique para procurar no seu computador (MP4, AVI, MOV, JPG, PNG)</p>
                </>
              )}
            </div>

            {/* Cloud API Toggle */}
            <div className="mt-6 flex items-center justify-center gap-3">
              <button
                onClick={(e) => { e.stopPropagation(); setUseCloudApi(!useCloudApi); }}
                className={`
                  relative flex items-center gap-3 px-5 py-2.5 rounded-full text-sm font-medium
                  transition-all duration-300 border
                  ${useCloudApi
                    ? 'bg-purple-500/15 border-purple-500/40 text-purple-300 shadow-[0_0_20px_-6px_rgba(168,85,247,0.4)]'
                    : isDark ? 'bg-zinc-900/50 border-zinc-700/50 text-zinc-500 hover:border-zinc-600' : 'bg-white/60 border-zinc-300 text-zinc-500 hover:border-zinc-400'}
                `}
              >
                {useCloudApi ? (
                  <Cloud className="w-4 h-4" />
                ) : (
                  <CloudOff className="w-4 h-4" />
                )}
                <span>{useCloudApi ? 'Cloud API Ativa' : 'Apenas Análise Local'}</span>
                <div className={`
                  w-9 h-5 rounded-full transition-colors duration-300 flex items-center px-0.5
                  ${useCloudApi ? 'bg-purple-500' : 'bg-zinc-700'}
                `}>
                  <div className={`
                    w-4 h-4 rounded-full bg-white shadow-md transition-transform duration-300
                    ${useCloudApi ? 'translate-x-4' : 'translate-x-0'}
                  `} />
                </div>
              </button>
            </div>

            {/* Error message */}
            {error && (
              <div
                className="mt-4 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm text-center"
                role="alert"
                aria-live="assertive"
              >
                ⚠️ {error}
              </div>
            )}

            {/* ── Progress Bar ─────────────────────────────────────────────── */}
            {isProcessing && progress && progress.active && (
              <div
                className="mt-6 w-full space-y-3"
                role="progressbar"
                aria-valuenow={progressPct}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuetext={progress.message}
                aria-live="polite"
              >
                <div className="flex justify-between items-center text-sm">
                  <span className="text-zinc-400 flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin text-purple-400" />
                    {progress.message}
                  </span>
                  {progress.stage === "analyzing" && (
                    <span className="text-purple-400 font-mono font-bold">
                      {progressPct}%
                    </span>
                  )}
                </div>

                <div className={`h-2 rounded-full overflow-hidden ${isDark ? 'bg-zinc-800' : 'bg-zinc-200'}`}>
                  {progress.stage === "analyzing" ? (
                    <div
                      className="h-full rounded-full bg-purple-500 transition-all duration-300 ease-out"
                      style={{ width: `${progressPct}%` }}
                    />
                  ) : (
                    <div className="h-full w-full progress-shimmer rounded-full" />
                  )}
                </div>

                {progress.stage === "analyzing" && (
                  <p className="text-xs text-zinc-500 text-center">
                    Frame {progress.current_frame} de {progress.total_frames}
                  </p>
                )}
              </div>
            )}

            <div className="mt-8 flex justify-center">
              <button
                disabled={!file || isProcessing}
                onClick={submitToEngine}
                className={`
                  relative overflow-hidden px-8 py-4 rounded-full font-bold transition-all
                  ${!file ? (isDark ? 'bg-zinc-800 text-zinc-500' : 'bg-zinc-200 text-zinc-400') + ' cursor-not-allowed' : 'bg-purple-600 text-white hover:bg-purple-500 hover:scale-105 active:scale-95 shadow-[0_0_40px_-10px_rgba(168,85,247,0.5)]'}
                `}
              >
                {isProcessing ? (
                  <span className="flex items-center gap-2">
                    <Loader2 className="w-5 h-5 animate-spin" /> A processar...
                  </span>
                ) : (
                  "Iniciar Análise"
                )}
              </button>
            </div>
          </div>
        )}

        {/* Results Dashboard — hidden in demo mode (no engine, no report possible). */}
        {!demoMode && report && (
          <div className="w-full">
            <div className="flex justify-center gap-3 mb-8">
              <button
                onClick={() => { setReport(null); setFile(null); setError(null); }}
                className="px-6 py-2 rounded-full glass text-sm font-medium hover:bg-white/10 transition-colors"
              >
                ← Analisar novo ficheiro
              </button>
              <button
                onClick={async () => {
                  const { generateForensicReport } = await import('@/utils/generateReport');
                  await generateForensicReport(report, {
                    apiBase: getApiBase(),
                    apiKey: API_KEY,
                  });
                }}
                className="px-6 py-2 rounded-full glass text-sm font-medium hover:bg-purple-500/20 hover:border-purple-500/40 transition-all flex items-center gap-2 border border-transparent"
              >
                <FileDown className="w-4 h-4 text-purple-400" />
                Exportar Relatório PDF
              </button>
            </div>
            <ReportDashboard report={report} />
          </div>
        )}

      </div>
    </main>
  );
}
