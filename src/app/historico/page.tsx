"use client";

import React, { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  Clock,
  Trash2,
  FileVideo,
  FileImage,
  AlertTriangle,
  RefreshCw,
  Inbox,
  Download as DownloadIcon,
} from "lucide-react";
import { useTheme } from "@/hooks/useTheme";
import TopNav from "@/components/TopNav";
import DemoModeBanner from "@/components/DemoModeBanner";
import { isDemoMode } from "@/utils/demoMode";

// ── Shape returned by GET /api/history ───────────────────────────────────
interface HistoryItem {
  id: string;
  filename: string;
  created_at: number; // unix seconds
  overall_score: number;
  verdict: "AUTÊNTICO" | "INCERTO" | "SUSPEITO" | string;
  is_image: boolean;
  frame_count: number;
  duration_secs: number | null;
  plugins: Record<string, number>;
  has_thumbnail: boolean;
}

interface HistoryResponse {
  items: HistoryItem[];
  total: number;
  limit: number;
  offset: number;
}

// ── API base — same convention as the rest of the frontend ───────────────
function getApiBase(): string {
  if (typeof window === "undefined") return "http://localhost:8000";
  const host = window.location.hostname === "localhost"
    ? "127.0.0.1"
    : window.location.hostname;
  return `http://${host}:8000`;
}

function getApiKey(): string | undefined {
  return process.env.NEXT_PUBLIC_ENGINE_API_KEY || undefined;
}

function authHeaders(): HeadersInit {
  const key = getApiKey();
  return key ? { "X-API-Key": key } : {};
}

// ── Display helpers ──────────────────────────────────────────────────────
function verdictColor(verdict: string): string {
  if (verdict === "SUSPEITO") return "text-red-400 border-red-500/50 bg-red-500/10";
  if (verdict === "INCERTO") return "text-yellow-400 border-yellow-500/50 bg-yellow-500/10";
  return "text-green-400 border-green-500/50 bg-green-500/10";
}

function formatDate(unixSecs: number): string {
  const d = new Date(unixSecs * 1000);
  return d.toLocaleString("pt-PT", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDuration(secs: number | null, isImage: boolean): string {
  if (isImage) return "imagem";
  if (secs == null) return "—";
  if (secs < 60) return `${secs.toFixed(1)}s`;
  const mins = Math.floor(secs / 60);
  const rem = Math.round(secs - mins * 60);
  return `${mins}m ${rem}s`;
}

// ── Page ────────────────────────────────────────────────────────────────
export default function HistoricoPage() {
  const { isDark } = useTheme();
  const [items, setItems] = useState<HistoryItem[] | null>(null);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  // Same SSR-safe demo flag as the detector page: defaults to true so the
  // public Vercel render shows the banner immediately, flips to false on
  // localhost after mount.
  const [demoMode, setDemoMode] = useState(true);
  useEffect(() => {
    setDemoMode(isDemoMode());
  }, []);

  const fetchHistory = useCallback(async () => {
    setRefreshing(true);
    setError(null);
    try {
      const res = await fetch(`${getApiBase()}/api/history?limit=200`, {
        headers: authHeaders(),
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data: HistoryResponse = await res.json();
      setItems(data.items);
      setTotal(data.total);
    } catch (e) {
      setError(
        e instanceof Error
          ? `Não foi possível carregar o histórico: ${e.message}`
          : "Não foi possível carregar o histórico."
      );
      setItems([]);
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    // Skip the fetch entirely on the public demo — the engine isn't there
    // and we'd just spam a 0.0.0.0 connection-refused error to the console.
    if (demoMode) return;
    fetchHistory();
  }, [fetchHistory, demoMode]);

  const handleDelete = async (id: string) => {
    if (!confirm("Eliminar esta análise do histórico?")) return;
    setDeletingId(id);
    try {
      const res = await fetch(`${getApiBase()}/api/history/${id}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setItems((prev) => prev?.filter((i) => i.id !== id) ?? null);
      setTotal((t) => Math.max(0, t - 1));
    } catch (e) {
      alert(
        e instanceof Error
          ? `Falha ao eliminar: ${e.message}`
          : "Falha ao eliminar."
      );
    } finally {
      setDeletingId(null);
    }
  };

  const handleClearAll = async () => {
    if (
      !confirm(
        "Apagar TODAS as análises do histórico? Esta operação é irreversível."
      )
    )
      return;
    setClearing(true);
    try {
      const res = await fetch(`${getApiBase()}/api/history`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setItems([]);
      setTotal(0);
    } catch (e) {
      alert(
        e instanceof Error
          ? `Falha ao limpar histórico: ${e.message}`
          : "Falha ao limpar histórico."
      );
    } finally {
      setClearing(false);
    }
  };

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

      {/* Demo mode: history lives in a local SQLite file that doesn't exist
          on the public deploy. Replace the entire viewer with the CTA banner
          instead of fetching against a non-existent engine. */}
      {demoMode && (
        <div className="relative z-10 max-w-5xl mx-auto px-6 pt-32 pb-20">
          <DemoModeBanner
            feature="Histórico de análises"
            reason="Cada análise é guardada numa base SQLite em ~/.deepfake-forensics/history.db na máquina onde o motor corre. Como o site público não tem motor (e nunca recebeu uploads), não há nada a listar aqui."
          />
        </div>
      )}

      {!demoMode && (
      <div className="relative z-10 max-w-5xl mx-auto px-6 pt-32 pb-20">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8">
          <div>
            <div
              className={`inline-flex items-center gap-2 px-3 py-1 rounded-full glass text-sm mb-4 border ${
                isDark
                  ? "text-zinc-300 border-zinc-800"
                  : "text-zinc-600 border-zinc-300"
              }`}
            >
              <Clock className="w-4 h-4 text-purple-400" />
              <span>Análises Anteriores</span>
            </div>
            <h1 className="text-3xl md:text-5xl font-black tracking-tight bg-gradient-to-br from-white via-white to-zinc-500 bg-clip-text text-transparent"
              style={!isDark ? { backgroundImage: "linear-gradient(to bottom right, #18181b, #18181b, #71717a)" } : undefined}
            >
              Histórico
            </h1>
            <p className={`mt-2 text-sm ${isDark ? "text-zinc-400" : "text-zinc-600"}`}>
              {total} análise{total === 1 ? "" : "s"} guardada{total === 1 ? "" : "s"} localmente em{" "}
              <code className={isDark ? "text-purple-400" : "text-purple-600"}>
                ~/.deepfake-forensics/history.db
              </code>
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={fetchHistory}
              disabled={refreshing}
              className={`inline-flex items-center gap-2 px-3 py-2 rounded-lg border text-sm transition-colors disabled:opacity-50 ${
                isDark
                  ? "border-zinc-700 hover:bg-zinc-900 text-zinc-200"
                  : "border-zinc-300 hover:bg-zinc-100 text-zinc-700"
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`} />
              {refreshing ? "A atualizar..." : "Atualizar"}
            </button>
            {items && items.length > 0 && (
              <button
                onClick={handleClearAll}
                disabled={clearing}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-red-500/50 hover:bg-red-500/10 text-red-400 text-sm transition-colors disabled:opacity-50"
              >
                <Trash2 className="w-4 h-4" />
                {clearing ? "A limpar..." : "Limpar tudo"}
              </button>
            )}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 flex items-start gap-3 p-4 rounded-lg border border-red-500/40 bg-red-500/10 text-sm text-red-300">
            <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
            <div>
              <p>{error}</p>
              <p className="mt-1 text-xs text-red-300/70">
                Verifica que o engine está a correr em <code>{getApiBase()}</code>.
              </p>
            </div>
          </div>
        )}

        {/* Loading skeleton */}
        {items === null && !error && (
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <div
                key={i}
                className={`rounded-xl border p-4 animate-pulse h-24 ${
                  isDark
                    ? "border-zinc-800 bg-zinc-900/50"
                    : "border-zinc-200 bg-zinc-100"
                }`}
              />
            ))}
          </div>
        )}

        {/* Empty state */}
        {items && items.length === 0 && !error && (
          <div
            className={`glass border rounded-2xl p-12 text-center ${
              isDark ? "border-zinc-800" : "border-zinc-200"
            }`}
          >
            <Inbox
              className={`w-12 h-12 mx-auto mb-4 ${
                isDark ? "text-zinc-600" : "text-zinc-400"
              }`}
            />
            <h2 className="text-lg font-semibold mb-2">
              Ainda não há análises guardadas
            </h2>
            <p
              className={`text-sm mb-6 ${
                isDark ? "text-zinc-400" : "text-zinc-600"
              }`}
            >
              Faz uma análise no detetor — fica automaticamente registada aqui.
            </p>
            <Link
              href="/"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium transition-colors"
            >
              Ir para o detetor
            </Link>
          </div>
        )}

        {/* List */}
        {items && items.length > 0 && (
          <div className="space-y-3">
            {items.map((item) => (
              <article
                key={item.id}
                className={`glass border rounded-xl overflow-hidden transition-colors ${
                  isDark
                    ? "border-zinc-800 hover:border-zinc-700"
                    : "border-zinc-200 hover:border-zinc-300"
                }`}
              >
                <div className="flex items-stretch gap-4 p-4">
                  {/* Thumbnail (clickable -> detail page) */}
                  <Link
                    href={`/historico/${item.id}`}
                    className={`relative shrink-0 w-32 h-20 rounded-lg overflow-hidden ${
                      isDark ? "bg-zinc-900" : "bg-zinc-200"
                    }`}
                  >
                    {item.has_thumbnail ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={`${getApiBase()}/api/history/${item.id}/thumbnail`}
                        alt={item.filename}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-zinc-500">
                        {item.is_image ? <FileImage className="w-6 h-6" /> : <FileVideo className="w-6 h-6" />}
                      </div>
                    )}
                  </Link>

                  {/* Info (also clickable) */}
                  <Link href={`/historico/${item.id}`} className="flex-1 min-w-0 block">
                    <div className="flex items-start justify-between gap-3">
                      <h3 className="font-semibold truncate" title={item.filename}>
                        {item.filename}
                      </h3>
                      <span
                        className={`shrink-0 px-2 py-0.5 rounded-md border text-xs font-bold ${verdictColor(
                          item.verdict
                        )}`}
                      >
                        {item.verdict} {(item.overall_score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div
                      className={`mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs ${
                        isDark ? "text-zinc-400" : "text-zinc-600"
                      }`}
                    >
                      <span>{formatDate(item.created_at)}</span>
                      <span>•</span>
                      <span>{item.frame_count} frame{item.frame_count === 1 ? "" : "s"}</span>
                      <span>•</span>
                      <span>{formatDuration(item.duration_secs, item.is_image)}</span>
                    </div>
                    {Object.keys(item.plugins).length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {Object.entries(item.plugins)
                          .sort((a, b) => b[1] - a[1])
                          .slice(0, 4)
                          .map(([name, score]) => (
                            <span
                              key={name}
                              className={`px-2 py-0.5 rounded text-[10px] font-mono ${
                                isDark
                                  ? "bg-zinc-800 text-zinc-300"
                                  : "bg-zinc-100 text-zinc-700"
                              }`}
                              title={`${name}: ${(score * 100).toFixed(1)}%`}
                            >
                              {name.split(" ")[0]} {(score * 100).toFixed(0)}%
                            </span>
                          ))}
                      </div>
                    )}
                  </Link>

                  {/* Actions (not part of the link) */}
                  <div className="flex flex-col items-end justify-between gap-2 shrink-0">
                    <button
                      onClick={() => handleDelete(item.id)}
                      disabled={deletingId === item.id}
                      title="Eliminar análise"
                      className="p-1.5 rounded-md text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-50"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                    <a
                      href={`${getApiBase()}/api/history/${item.id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      title="Ver JSON completo da análise (debug)"
                      className={`p-1.5 rounded-md transition-colors ${
                        isDark
                          ? "text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800"
                          : "text-zinc-500 hover:text-zinc-800 hover:bg-zinc-100"
                      }`}
                    >
                      <DownloadIcon className="w-4 h-4" />
                    </a>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>
      )}
    </main>
  );
}
