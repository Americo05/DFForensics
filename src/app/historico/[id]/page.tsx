"use client";

import React, { use, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  Trash2,
  AlertTriangle,
  FileVideo,
  FileImage,
  Clock,
  Cpu,
} from "lucide-react";
import { useTheme } from "@/hooks/useTheme";
import TopNav from "@/components/TopNav";

// ── Types mirroring history_store row shape ─────────────────────────────
interface FaceDetail {
  face_bbox?: { x: number; y: number; w: number; h: number } | null;
  scene_detected?: string;
  overall_score?: number;
  plugin_scores?: Record<string, number>;
}

interface FrameDetail {
  frame_index: number;
  timestamp_seconds?: number;
  overall_score: number;
  faces?: FaceDetail[];
}

interface HistoryDetail {
  id: string;
  filename: string;
  created_at: number;
  overall_score: number;
  verdict: string;
  is_image: boolean;
  frame_count: number;
  duration_secs: number | null;
  plugins: Record<string, number>;
  has_thumbnail: boolean;
  frame_details: FrameDetail[];
}

// ── API helpers ──────────────────────────────────────────────────────────
function getApiBase(): string {
  if (typeof window === "undefined") return "http://localhost:8000";
  const host = window.location.hostname === "localhost"
    ? "127.0.0.1"
    : window.location.hostname;
  return `http://${host}:8000`;
}

function authHeaders(): HeadersInit {
  const key = process.env.NEXT_PUBLIC_ENGINE_API_KEY;
  return key ? { "X-API-Key": key } : {};
}

// ── Display helpers ──────────────────────────────────────────────────────
function verdictColor(verdict: string): string {
  if (verdict === "SUSPEITO") return "text-red-400 border-red-500/50 bg-red-500/10";
  if (verdict === "INCERTO") return "text-yellow-400 border-yellow-500/50 bg-yellow-500/10";
  return "text-green-400 border-green-500/50 bg-green-500/10";
}

function scoreToColor(score: number): string {
  const r = Math.round(Math.min(255, score * 2 * 255));
  const g = Math.round(Math.min(255, (1 - score) * 2 * 255));
  return `rgb(${r},${g},40)`;
}

function formatDate(unixSecs: number): string {
  return new Date(unixSecs * 1000).toLocaleString("pt-PT", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDuration(secs: number | null, isImage: boolean): string {
  if (isImage) return "imagem (frame único)";
  if (secs == null) return "duração desconhecida";
  if (secs < 60) return `${secs.toFixed(1)} segundos`;
  const mins = Math.floor(secs / 60);
  const rem = Math.round(secs - mins * 60);
  return `${mins}m ${rem}s`;
}

// ── Page ────────────────────────────────────────────────────────────────
// Next.js 15 App Router: params is now a Promise that must be unwrapped with React.use().
export default function HistoricoDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const router = useRouter();
  const { isDark } = useTheme();
  const [data, setData] = useState<HistoryDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const fetchDetail = useCallback(async () => {
    setError(null);
    try {
      const res = await fetch(`${getApiBase()}/api/history/${id}`, {
        headers: authHeaders(),
      });
      if (res.status === 404) {
        setError("Análise não encontrada — pode ter sido eliminada.");
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = (await res.json()) as HistoryDetail;
      setData(json);
    } catch (e) {
      setError(
        e instanceof Error
          ? `Falha a carregar análise: ${e.message}`
          : "Falha a carregar análise."
      );
    }
  }, [id]);

  useEffect(() => {
    fetchDetail();
  }, [fetchDetail]);

  const handleDelete = async () => {
    if (!confirm("Eliminar esta análise do histórico? Não é reversível.")) return;
    setDeleting(true);
    try {
      const res = await fetch(`${getApiBase()}/api/history/${id}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      router.push("/historico");
    } catch (e) {
      alert(
        e instanceof Error ? `Falha ao eliminar: ${e.message}` : "Falha ao eliminar."
      );
      setDeleting(false);
    }
  };

  // Sort plugins by score descending — most suspicious at the top.
  const sortedPlugins = data
    ? Object.entries(data.plugins).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <main
      className={`min-h-screen selection:bg-purple-500/30 transition-colors duration-300 ${
        isDark ? "bg-[#050505] text-white" : "bg-[#f8f9fb] text-zinc-900"
      }`}
    >
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-purple-600/20 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-600/10 blur-[120px] rounded-full" />
      </div>

      <TopNav />

      <div className="relative z-10 max-w-5xl mx-auto px-6 pt-32 pb-20">
        <Link
          href="/historico"
          className={`inline-flex items-center gap-2 text-sm mb-6 transition-colors ${
            isDark ? "text-zinc-400 hover:text-white" : "text-zinc-600 hover:text-zinc-900"
          }`}
        >
          <ArrowLeft className="w-4 h-4" />
          Voltar ao histórico
        </Link>

        {/* Error */}
        {error && (
          <div className="flex items-start gap-3 p-4 rounded-lg border border-red-500/40 bg-red-500/10 text-sm text-red-300">
            <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
            <div>
              <p>{error}</p>
              <p className="mt-1 text-xs text-red-300/70">
                Engine em <code>{getApiBase()}</code>?
              </p>
            </div>
          </div>
        )}

        {/* Loading skeleton */}
        {!data && !error && (
          <div className="space-y-4">
            <div
              className={`rounded-2xl border p-8 animate-pulse h-48 ${
                isDark ? "border-zinc-800 bg-zinc-900/50" : "border-zinc-200 bg-zinc-100"
              }`}
            />
            <div
              className={`rounded-2xl border p-8 animate-pulse h-64 ${
                isDark ? "border-zinc-800 bg-zinc-900/50" : "border-zinc-200 bg-zinc-100"
              }`}
            />
          </div>
        )}

        {/* Detail */}
        {data && (
          <div className="space-y-6">
            {/* Hero card */}
            <section
              className={`glass border rounded-2xl overflow-hidden ${
                isDark ? "border-zinc-800" : "border-zinc-200"
              }`}
            >
              <div className="grid grid-cols-1 md:grid-cols-[280px_1fr]">
                {/* Thumbnail */}
                <div
                  className={`relative aspect-video md:aspect-auto ${
                    isDark ? "bg-zinc-900" : "bg-zinc-200"
                  }`}
                >
                  {data.has_thumbnail ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={`${getApiBase()}/api/history/${data.id}/thumbnail`}
                      alt={data.filename}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-zinc-500">
                      {data.is_image ? (
                        <FileImage className="w-12 h-12" />
                      ) : (
                        <FileVideo className="w-12 h-12" />
                      )}
                    </div>
                  )}
                </div>

                {/* Header info */}
                <div className="p-6 flex flex-col justify-between gap-4">
                  <div>
                    <h1 className="text-2xl font-bold truncate" title={data.filename}>
                      {data.filename}
                    </h1>
                    <p
                      className={`mt-1 text-xs ${
                        isDark ? "text-zinc-500" : "text-zinc-500"
                      }`}
                    >
                      ID: <code>{data.id}</code>
                    </p>

                    <div className="mt-4 flex items-center gap-3">
                      <span
                        className={`px-3 py-1 rounded-md border text-sm font-bold ${verdictColor(
                          data.verdict
                        )}`}
                      >
                        {data.verdict}
                      </span>
                      <div className="flex items-baseline gap-1">
                        <span
                          className="text-3xl font-black"
                          style={{ color: scoreToColor(data.overall_score) }}
                        >
                          {(data.overall_score * 100).toFixed(1)}%
                        </span>
                        <span
                          className={`text-xs ${
                            isDark ? "text-zinc-500" : "text-zinc-500"
                          }`}
                        >
                          probabilidade de manipulação
                        </span>
                      </div>
                    </div>
                  </div>

                  <div
                    className={`grid grid-cols-3 gap-3 text-xs ${
                      isDark ? "text-zinc-400" : "text-zinc-600"
                    }`}
                  >
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <Clock className="w-3 h-3" />
                        <span className="uppercase tracking-wider">Análise</span>
                      </div>
                      <p
                        className={isDark ? "text-zinc-200" : "text-zinc-900"}
                      >
                        {formatDate(data.created_at)}
                      </p>
                    </div>
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <Cpu className="w-3 h-3" />
                        <span className="uppercase tracking-wider">Frames</span>
                      </div>
                      <p
                        className={isDark ? "text-zinc-200" : "text-zinc-900"}
                      >
                        {data.frame_count}
                      </p>
                    </div>
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <FileVideo className="w-3 h-3" />
                        <span className="uppercase tracking-wider">Duração</span>
                      </div>
                      <p
                        className={isDark ? "text-zinc-200" : "text-zinc-900"}
                      >
                        {formatDuration(data.duration_secs, data.is_image)}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleDelete}
                      disabled={deleting}
                      className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-red-500/50 hover:bg-red-500/10 text-red-400 text-xs transition-colors disabled:opacity-50"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      {deleting ? "A eliminar..." : "Eliminar"}
                    </button>
                    <a
                      href={`${getApiBase()}/api/history/${data.id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs transition-colors ${
                        isDark
                          ? "border-zinc-700 hover:bg-zinc-900 text-zinc-200"
                          : "border-zinc-300 hover:bg-zinc-100 text-zinc-700"
                      }`}
                    >
                      Ver JSON cru
                    </a>
                  </div>
                </div>
              </div>
            </section>

            {/* Plugin scores */}
            <section
              className={`glass border rounded-2xl p-6 ${
                isDark ? "border-zinc-800" : "border-zinc-200"
              }`}
            >
              <h2 className="text-lg font-bold mb-4">Scores por detetor</h2>
              {sortedPlugins.length === 0 ? (
                <p className={`text-sm ${isDark ? "text-zinc-500" : "text-zinc-500"}`}>
                  Nenhum detetor registou scores nesta análise.
                </p>
              ) : (
                <div className="space-y-3">
                  {sortedPlugins.map(([name, score]) => {
                    const pct = (score * 100).toFixed(1);
                    const color = scoreToColor(score);
                    return (
                      <div key={name}>
                        <div className="flex justify-between items-center mb-1">
                          <span
                            className={`text-sm ${
                              isDark ? "text-zinc-200" : "text-zinc-800"
                            }`}
                          >
                            {name}
                          </span>
                          <span
                            className="text-sm font-mono font-bold"
                            style={{ color }}
                          >
                            {pct}%
                          </span>
                        </div>
                        <div
                          className={`h-2 rounded-full overflow-hidden ${
                            isDark ? "bg-zinc-800" : "bg-zinc-200"
                          }`}
                        >
                          <div
                            className="h-full rounded-full transition-all duration-700"
                            style={{ width: `${pct}%`, background: color }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>

            {/* Frame timeline */}
            {data.frame_details.length > 0 && (
              <section
                className={`glass border rounded-2xl p-6 ${
                  isDark ? "border-zinc-800" : "border-zinc-200"
                }`}
              >
                <h2 className="text-lg font-bold mb-1">
                  Timeline de suspeição por frame
                </h2>
                <p
                  className={`text-xs mb-4 ${
                    isDark ? "text-zinc-500" : "text-zinc-500"
                  }`}
                >
                  Cada barra representa o score combinado num frame. Frames
                  individuais não estão guardados (apenas a thumbnail) — esta
                  vista mostra a tendência ao longo do tempo.
                </p>
                <div className="flex gap-0.5 items-end h-32">
                  {data.frame_details.map((f) => {
                    const pct = Math.max(2, f.overall_score * 100);
                    return (
                      <div
                        key={f.frame_index}
                        className="flex-1 min-w-[3px] rounded-t transition-colors"
                        style={{
                          height: `${pct}%`,
                          background: scoreToColor(f.overall_score),
                        }}
                        title={`Frame #${f.frame_index + 1} — ${(
                          f.overall_score * 100
                        ).toFixed(1)}%`}
                      />
                    );
                  })}
                </div>
                <div
                  className={`mt-3 flex items-center gap-2 text-xs ${
                    isDark ? "text-zinc-500" : "text-zinc-500"
                  }`}
                >
                  <div
                    className="h-2 w-6 rounded"
                    style={{ background: "rgb(40, 255, 40)" }}
                  />
                  <span>Autêntico</span>
                  <div
                    className="h-2 w-6 rounded ml-2"
                    style={{ background: "rgb(255, 255, 40)" }}
                  />
                  <span>Incerto</span>
                  <div
                    className="h-2 w-6 rounded ml-2"
                    style={{ background: "rgb(255, 40, 40)" }}
                  />
                  <span>Suspeito</span>
                </div>
              </section>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
