"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { Eye, Info, ZoomIn } from "lucide-react";
import type { FrameDetail } from "@/types/report";

interface VideoForensicsPlayerProps {
  analysisId: string;
  frameDetails: FrameDetail[];
  pluginNames: string[];
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function scoreToColor(score: number): string {
  // 0.0 → green,  0.5 → yellow,  1.0 → red
  const r = Math.round(Math.min(255, score * 2 * 255));
  const g = Math.round(Math.min(255, (1 - score) * 2 * 255));
  return `rgb(${r},${g},40)`;
}

function scoreLabel(score: number): string {
  if (score >= 0.75) return "SUSPEITO";
  if (score >= 0.5) return "INCERTO";
  return "AUTÊNTICO";
}

function getApiBase(): string {
  if (typeof window === "undefined") return "http://localhost:8000";
  const host = window.location.hostname === "localhost" ? "127.0.0.1" : window.location.hostname;
  return `http://${host}:8000`;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function VideoForensicsPlayer({
  analysisId,
  frameDetails,
  pluginNames,
}: VideoForensicsPlayerProps) {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  // Tracks frame indices whose image fetch returned 404 / decode failed, so
  // we render a placeholder instead of a broken-image icon. Set instead of
  // boolean so we don't have to reset on every navigation.
  const [erroredFrames, setErroredFrames] = useState<Set<number>>(new Set());
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const selected = frameDetails[selectedIdx];

  // Compute the frame URL during render instead of mirroring it into a
  // useEffect → setState pair, which trips react-hooks/set-state-in-effect.
  // Same value, no extra render cycle, no rule violation.
  const imgSrc =
    analysisId && frameDetails.length > 0 && !erroredFrames.has(selectedIdx)
      ? `${getApiBase()}/api/frame/${analysisId}/${selectedIdx}`
      : "";

  // Draw bounding box on canvas whenever image loads or selection changes
  const drawOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const { naturalWidth, naturalHeight, offsetWidth, offsetHeight } = img;
    if (!naturalWidth || !offsetWidth) return;

    canvas.width = offsetWidth;
    canvas.height = offsetHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!selected?.faces) return;

    selected.faces.forEach((face) => {
      if (!face.face_bbox) return;

      // Scale bbox from natural image dimensions to displayed dimensions
      const scaleX = offsetWidth / naturalWidth;
      const scaleY = offsetHeight / naturalHeight;
      const { x, y, w, h } = face.face_bbox;
      const bx = x * scaleX;
      const by = y * scaleY;
      const bw = w * scaleX;
      const bh = h * scaleY;

      const color = scoreToColor(face.overall_score);

      // Outer box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;
      ctx.strokeRect(bx, by, bw, bh);

      // Corner accents
      const cs = 14; // corner size
      ctx.lineWidth = 4;
      [
        [bx, by, cs, 0, 0, cs],
        [bx + bw, by, -cs, 0, 0, cs],
        [bx, by + bh, cs, 0, 0, -cs],
        [bx + bw, by + bh, -cs, 0, 0, -cs],
      ].forEach(([cx, cy, dx1, dy1, dx2, dy2]) => {
        ctx.beginPath();
        ctx.moveTo(cx + dx1, cy + dy1);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx + dx2, cy + dy2);
        ctx.stroke();
      });

      // Label pill above box
      const label = `${scoreLabel(face.overall_score)} ${(face.overall_score * 100).toFixed(1)}%`;
      ctx.shadowBlur = 0;
      ctx.font = "bold 12px Inter, sans-serif";
      const textW = ctx.measureText(label).width;
      const pillX = bx;
      const pillY = by - 26;
      const pillH = 20;
      const pillW = textW + 16;

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(pillX, Math.max(0, pillY), pillW, pillH, 4);
      ctx.fill();

      ctx.fillStyle = "#fff";
      ctx.fillText(label, pillX + 8, Math.max(14, pillY + 14));
    });
  }, [selected]);

  useEffect(() => {
    drawOverlay();
  }, [drawOverlay, imgSrc, imgSize]);

  const handleImgLoad = () => {
    if (imgRef.current) {
      setImgSize({ w: imgRef.current.offsetWidth, h: imgRef.current.offsetHeight });
    }
    drawOverlay();
  };

  if (!frameDetails || frameDetails.length === 0) return null;

  return (
    <div className="w-full space-y-4">
      {/* Title */}
      <div className="flex items-center gap-2">
        <Eye className="w-5 h-5 text-purple-400" />
        <h2 className="text-xl font-bold text-white">Visualizador Forense por Frame</h2>
      </div>

      {/* Main grid: frame preview + plugin scores */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* Frame preview with canvas overlay */}
        <div className="lg:col-span-2 relative flex items-center justify-center rounded-xl overflow-hidden bg-zinc-900 border border-zinc-800 min-h-[320px]">
          {imgSrc ? (
            <div className="relative inline-block h-full max-h-[320px]">
              {/* The src is a runtime URL from our own API; using next/image
                  would force us to register the API host in next.config.ts
                  remotePatterns AND provide static width/height. Plain <img>
                  is the pragmatic choice for backend-streamed frames. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                ref={imgRef}
                src={imgSrc}
                alt={`Frame ${selectedIdx}`}
                className="block max-h-[320px] max-w-full object-contain"
                onLoad={handleImgLoad}
                onError={() => setErroredFrames((prev) => new Set(prev).add(selectedIdx))}
              />
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full pointer-events-none"
              />
            </div>
          ) : (
            <div className="flex items-center justify-center text-zinc-500">
              <ZoomIn className="w-8 h-8 mr-2" />
              <span>Seleciona um frame na timeline</span>
            </div>
          )}

          {/* Frame metadata bar */}
          {selected && (
            <div className="absolute bottom-0 left-0 right-0 bg-black/70 backdrop-blur-sm px-3 py-2 flex justify-between items-center text-xs text-zinc-300">
              <span className="font-mono">Frame #{selectedIdx + 1}</span>
              <span className="font-mono">{selected.timestamp_seconds.toFixed(2)}s</span>
              <span
                className="font-bold px-2 py-0.5 rounded"
                style={{ color: scoreToColor(selected.overall_score), border: `1px solid ${scoreToColor(selected.overall_score)}` }}
              >
                {(selected.overall_score * 100).toFixed(1)}%
              </span>
            </div>
          )}
        </div>

        {/* Per-face plugin scores panel */}
        <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4 space-y-4 max-h-[320px] overflow-y-auto">
          <div className="flex items-center gap-2 mb-2 sticky top-0 bg-zinc-900 z-10 py-1">
            <Info className="w-4 h-4 text-purple-400" />
            <span className="text-sm font-bold text-white">
              Análise de {selected?.faces?.length || 0} pessoa{(selected?.faces?.length !== 1) ? 's' : ''}
            </span>
          </div>
          
          {selected?.faces?.map((face, faceIdx) => (
            <div key={faceIdx} className="space-y-3 bg-zinc-800/50 p-3 rounded-lg border border-zinc-800">
              <div className="flex justify-between items-center border-b border-zinc-700/50 pb-2">
                <p className="text-xs text-zinc-300 font-bold">
                  👤 Sujeito #{faceIdx + 1}
                </p>
                <span
                  className="px-2 py-0.5 rounded text-[10px] font-bold"
                  style={{ color: scoreToColor(face.overall_score), border: `1px solid ${scoreToColor(face.overall_score)}` }}
                >
                  {scoreLabel(face.overall_score)} ({(face.overall_score * 100).toFixed(1)}%)
                </span>
              </div>
              
              {pluginNames.map((name) => {
                let score = face.plugin_scores?.[name];
                if (typeof score !== 'number' || isNaN(score as number)) score = 0;
                
                const pct = (Number(score) * 100).toFixed(1);
                const color = scoreToColor(Number(score));
                return (
                  <div key={name} className="space-y-1">
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-zinc-400 truncate max-w-[70%]" title={name}>
                        {name.replace(/ Detector| Analyser/gi, "")}
                      </span>
                      <span className="text-[10px] font-bold" style={{ color }}>{pct}%</span>
                    </div>
                    <div className="h-1 rounded-full bg-zinc-700 overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{ width: `${pct}%`, background: color }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ))}

          {(!selected?.faces || selected.faces.length === 0) && (
            <p className="text-xs text-zinc-600 pt-2 border-t border-zinc-800">
              ⚠️ Nenhuma cara detetada neste frame
            </p>
          )}
        </div>
      </div>

      {/* Frame Timeline */}
      <div className="rounded-xl bg-zinc-900 border border-zinc-800 p-4">
        <p className="text-xs text-zinc-500 mb-3">
          Timeline de Suspeição — clica num frame para inspecioná-lo
        </p>
        <div className="flex gap-1 flex-wrap">
          {frameDetails.map((f, i) => {
            const isSelected = i === selectedIdx;
            const bg = scoreToColor(f.overall_score);
            return (
              <button
                key={i}
                title={`Frame ${i + 1} — ${(f.overall_score * 100).toFixed(1)}% @ ${f.timestamp_seconds.toFixed(2)}s`}
                onClick={() => setSelectedIdx(i)}
                className={`relative h-10 flex-1 min-w-[28px] max-w-[60px] rounded transition-all duration-200 ${isSelected ? "ring-2 ring-white scale-110 z-10" : "opacity-80 hover:opacity-100 hover:scale-105"
                  }`}
                style={{ background: bg }}
              >
                <span className="absolute bottom-0.5 left-0 right-0 text-center text-[9px] text-white/80 font-mono leading-tight">
                  F{i + 1}
                </span>
              </button>
            );
          })}
        </div>

        {/* Score legend */}
        <div className="flex items-center gap-2 mt-3 text-xs text-zinc-500">
          <div className="h-2 w-8 rounded" style={{ background: "rgb(40, 255, 40)" }} />
          <span>Autêntico</span>
          <div className="h-2 w-8 rounded ml-2" style={{ background: "rgb(255, 255, 40)" }} />
          <span>Incerto</span>
          <div className="h-2 w-8 rounded ml-2" style={{ background: "rgb(255, 40, 40)" }} />
          <span>Suspeito</span>
        </div>
      </div>
    </div>
  );
}
