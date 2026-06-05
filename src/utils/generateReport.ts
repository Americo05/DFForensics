/**
 * generateReport.ts — Forensic PDF Report Generator
 *
 * Generates a professional, detailed PDF report from analysis results.
 * Adapts content based on analysis mode (Cloud API vs Local).
 *
 * Dependencies: jspdf, jspdf-autotable
 */

import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import type { AnalysisReport, PluginResult } from "@/types/report";
import { classifyScore } from "@/utils/verdict";

// jsPDF subtype that adds the autotable handle. Keeps the cast at the boundary.
type DocWithAutoTable = jsPDF & { lastAutoTable?: { finalY: number } };

type ReportData = AnalysisReport;

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function pct(score: number): string {
  return `${(score * 100).toFixed(1)}%`;
}

function verdictText(score: number): string {
  return classifyScore(score).label;
}

function verdictDescription(score: number): string {
  return classifyScore(score).description;
}

function modeLabel(mode?: string): string {
  switch (mode) {
    case "cloud": return "Cloud API (Sightengine)";
    case "local": return "Análise Local (ViT + DCT + Edge Blending)";
    default:      return "Todos os Plugins";
  }
}

function now(): string {
  return new Date().toLocaleString("pt-PT", {
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

// ── Colors ───────────────────────────────────────────────────────────────────

const COLORS = {
  bg:       [9, 9, 11] as [number, number, number],
  card:     [24, 24, 27] as [number, number, number],
  accent:   [147, 51, 234] as [number, number, number],  // purple-600
  green:    [34, 197, 94] as [number, number, number],
  red:      [239, 68, 68] as [number, number, number],
  yellow:   [234, 179, 8] as [number, number, number],
  white:    [250, 250, 250] as [number, number, number],
  muted:    [161, 161, 170] as [number, number, number],
  dimmed:   [113, 113, 122] as [number, number, number],
  border:   [39, 39, 42] as [number, number, number],
};

// ── Top-frames thumbnail fetch ───────────────────────────────────────────
// Fetch top-N most suspicious frames as base64 JPEGs so jsPDF can embed
// them with doc.addImage(). Network failures degrade gracefully — the
// section is just omitted when no thumbs can be loaded.

interface FrameThumb {
  index: number;
  score: number;
  timestamp: number;
  dataUrl: string;
}

async function fetchFrameThumb(
  apiBase: string,
  analysisId: string,
  frameIndex: number,
  apiKey: string | undefined,
): Promise<string | null> {
  try {
    const headers: HeadersInit = apiKey ? { 'X-API-Key': apiKey } : {};
    const res = await fetch(`${apiBase}/api/frame/${encodeURIComponent(analysisId)}/${frameIndex}`, { headers });
    if (!res.ok) return null;
    const blob = await res.blob();
    return await new Promise<string | null>((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(typeof reader.result === 'string' ? reader.result : null);
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
}

async function fetchTopSuspiciousFrames(
  data: ReportData,
  apiBase: string | undefined,
  apiKey: string | undefined,
  limit = 3,
): Promise<FrameThumb[]> {
  if (!apiBase || !data.results.frame_details) return [];
  const sorted = [...data.results.frame_details]
    .map((f, i) => ({ idx: i, score: f.overall_score, ts: f.timestamp_seconds }))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);

  const thumbs: FrameThumb[] = [];
  for (const item of sorted) {
    const dataUrl = await fetchFrameThumb(apiBase, data.analysis_id, item.idx, apiKey);
    if (dataUrl) {
      thumbs.push({ index: item.idx, score: item.score, timestamp: item.ts, dataUrl });
    }
  }
  return thumbs;
}

// ── Main Generator ──────────────────────────────────────────────────────────

export async function generateForensicReport(
  data: ReportData,
  options: { apiBase?: string; apiKey?: string } = {},
): Promise<void> {
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const pageW = doc.internal.pageSize.getWidth();
  const margin = 15;
  const contentW = pageW - margin * 2;
  let y = 0;

  const { metadata, results, lip_sync, audio_deepfake } = data;
  const overallScore = typeof results.overall_score === 'number' ? results.overall_score : 0;
  const overallVerdict = classifyScore(overallScore);
  const isImage = metadata.format === "image";
  const docWithTable = doc as DocWithAutoTable;

  // ── Page Background ─────────────────────────────────────────────────────
  function drawBackground() {
    doc.setFillColor(...COLORS.bg);
    doc.rect(0, 0, pageW, doc.internal.pageSize.getHeight(), "F");
  }

  function ensureSpace(needed: number) {
    if (y + needed > doc.internal.pageSize.getHeight() - 15) {
      doc.addPage();
      drawBackground();
      y = 15;
    }
  }

  drawBackground();

  // ── Header ──────────────────────────────────────────────────────────────
  y = 15;
  doc.setFillColor(...COLORS.accent);
  doc.roundedRect(margin, y, contentW, 28, 3, 3, "F");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.setTextColor(...COLORS.white);
  doc.text("RELATÓRIO DE ANÁLISE FORENSE", margin + 8, y + 11);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(220, 220, 230);
  doc.text(`Motor de Deteção Forense v2.0 — Sistema de Plugins`, margin + 8, y + 18);
  doc.text(`Gerado em: ${now()}`, margin + 8, y + 24);
  doc.text(`ID: ${data.analysis_id}`, pageW - margin - 8, y + 24, { align: "right" });

  y += 35;

  // ── Section: File Metadata ──────────────────────────────────────────────
  ensureSpace(40);
  doc.setFillColor(...COLORS.card);
  doc.roundedRect(margin, y, contentW, isImage ? 28 : 38, 2, 2, "F");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(...COLORS.accent);
  doc.text("FICHEIRO ANALISADO", margin + 6, y + 8);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(...COLORS.white);

  const col1 = margin + 6;
  const col2 = margin + contentW / 2;

  doc.text(`Nome: ${metadata.filename}`, col1, y + 16);
  doc.text(`Tamanho: ${formatBytes(metadata.filesize_bytes)}`, col2, y + 16);

  if (!isImage) {
    doc.text(`Resolução: ${metadata.resolution || "N/A"} · ${metadata.fps?.toFixed(0) || "?"} fps`, col1, y + 23);
    doc.text(`Duração: ${metadata.duration_seconds?.toFixed(1) || "?"}s`, col2, y + 23);
    doc.text(`Frames analisados: ${metadata.frames_analyzed} de ${metadata.total_frames || "?"}`, col1, y + 30);
    doc.text(`Método: ${metadata.extraction_method || "N/A"}`, col2, y + 30);
    y += 44;
  } else {
    doc.text(`Formato: Imagem`, col1, y + 23);
    y += 34;
  }

  // ── Section: Verdict ────────────────────────────────────────────────────
  ensureSpace(38);
  const verdictColor = overallVerdict.rgb;

  doc.setFillColor(...COLORS.card);
  doc.roundedRect(margin, y, contentW, 32, 2, 2, "F");

  // Colored left border
  doc.setFillColor(...verdictColor);
  doc.rect(margin, y, 3, 32, "F");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(...COLORS.accent);
  doc.text("VEREDITO", margin + 10, y + 8);

  doc.setFont("helvetica", "bold");
  doc.setFontSize(14);
  doc.setTextColor(...verdictColor);
  doc.text(verdictText(overallScore), margin + 10, y + 18);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(...COLORS.muted);
  doc.text(verdictDescription(overallScore), margin + 10, y + 24);

  // "Triggered by" — only useful when the verdict actually IS fake, otherwise
  // it just says "visual won because it's the only modality" on a clean clip.
  if (overallVerdict.isFake && results.triggered_by) {
    const label = results.triggered_by === 'visual' ? 'Visual (Face/Píxeis)'
                : results.triggered_by === 'wavlm' ? 'Áudio (Clonagem Voz)'
                : 'Áudio (Sincronia Labial)';
    doc.setFontSize(7);
    doc.text(`Sinal dominante: ${label}`, margin + 10, y + 29);
  }

  // Scores on the right
  doc.setFontSize(9);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(...COLORS.white);

  const visualScore = typeof results.visual_score === 'number' ? results.visual_score : overallScore;
  doc.text(`Risco Visual: ${pct(visualScore)}`, col2, y + 12);

  if (results.audio_score != null) {
    doc.text(`Risco de Áudio: ${pct(results.audio_score)}`, col2, y + 19);
  } else {
    doc.setTextColor(...COLORS.dimmed);
    doc.text(`Risco de Áudio: Sem dados`, col2, y + 19);
  }

  doc.setTextColor(...COLORS.muted);
  doc.setFontSize(8);
  doc.text(`Modo: ${modeLabel(metadata.analysis_mode)}`, col2, y + 26);

  y += 38;

  // ── Section: Plugin Breakdown Table ─────────────────────────────────────
  if (results.plugins && results.plugins.length > 0) {
    ensureSpace(20 + results.plugins.length * 8);

    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(...COLORS.accent);
    doc.text("ANÁLISE POR PLUGIN", margin + 6, y + 8);
    y += 12;

    // Build the rows with a stable index → plugin lookup so the cell coloring
    // hook can find the right plugin even if the row order ever changes.
    const pluginRowEntries: { plugin: PluginResult; row: (string | number)[] }[] =
      results.plugins.map((p) => ({
        plugin: p,
        row: [
          p.name,
          pct(p.average_score),
          p.frames_analyzed.toString(),
          classifyScore(p.average_score).label,
        ],
      }));

    autoTable(doc, {
      startY: y,
      margin: { left: margin, right: margin },
      head: [["Plugin", "Score Médio", "Deteções", "Estado"]],
      body: pluginRowEntries.map((e) => e.row),
      theme: "plain",
      styles: {
        fillColor: COLORS.card,
        textColor: COLORS.white,
        fontSize: 9,
        cellPadding: 3,
        lineColor: COLORS.border,
        lineWidth: 0.3,
      },
      headStyles: {
        fillColor: [39, 39, 42],
        textColor: COLORS.accent,
        fontStyle: "bold",
        fontSize: 9,
      },
      columnStyles: {
        1: { halign: "center", fontStyle: "bold" },
        2: { halign: "center" },
        3: { halign: "center" },
      },
      didParseCell: (hookData) => {
        // Color the score and state columns based on the plugin for THIS row
        if (hookData.section === "body") {
          const entry = pluginRowEntries[hookData.row.index];
          if (entry && (hookData.column.index === 1 || hookData.column.index === 3)) {
            hookData.cell.styles.textColor = classifyScore(entry.plugin.average_score).rgb;
          }
        }
      },
    });

    y = (docWithTable.lastAutoTable?.finalY ?? y) + 6;
  }

  // ── Section: Audio-Visual Analysis ──────────────────────────────────────
  if (audio_deepfake || lip_sync) {
    ensureSpace(45);

    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(...COLORS.accent);
    doc.text("ANÁLISE ÁUDIO-VISUAL", margin + 6, y + 8);
    y += 14;

    doc.setFillColor(...COLORS.card);
    const audioBoxH = (audio_deepfake && lip_sync) ? 36 : 20;
    doc.roundedRect(margin, y, contentW, audioBoxH, 2, 2, "F");

    doc.setFontSize(9);
    let audioY = y + 8;

    if (audio_deepfake) {
      const aColor = classifyScore(audio_deepfake.audio_fake_score).rgb;
      doc.setFont("helvetica", "bold");
      doc.setTextColor(...COLORS.white);
      doc.text("Clonagem de Voz (WavLM)", col1, audioY);
      doc.setTextColor(...aColor);
      doc.text(pct(audio_deepfake.audio_fake_score), col2, audioY);

      doc.setFont("helvetica", "normal");
      doc.setTextColor(...COLORS.muted);
      doc.text(`Veredito: ${audio_deepfake.verdict}`, col2 + 30, audioY);
      audioY += 8;
    }

    if (lip_sync) {
      if (lip_sync.inconclusive) {
        doc.setFont("helvetica", "bold");
        doc.setTextColor(...COLORS.white);
        doc.text("Sincronia Labial (Lip Sync)", col1, audioY);
        doc.setTextColor(...COLORS.yellow);
        doc.text("INCONCLUSIVO", col2, audioY);

        doc.setFont("helvetica", "normal");
        doc.setTextColor(...COLORS.muted);
        audioY += 7;
        doc.text("Dados insuficientes para medir correlação.", col1, audioY);
      } else {
        const lColor = classifyScore(lip_sync.lip_sync_score).rgb;
        doc.setFont("helvetica", "bold");
        doc.setTextColor(...COLORS.white);
        doc.text("Sincronia Labial (Lip Sync)", col1, audioY);
        doc.setTextColor(...lColor);
        doc.text(pct(lip_sync.lip_sync_score), col2, audioY);

        doc.setFont("helvetica", "normal");
        doc.setTextColor(...COLORS.muted);
        audioY += 7;
        doc.text(
          `Correlação: ${lip_sync.correlation.toFixed(3)} · ${lip_sync.frames_analyzed} frames · ${lip_sync.verdict}`,
          col1, audioY
        );
      }
    }

    y += audioBoxH + 6;
  }

  // ── Section: Frame Statistics ───────────────────────────────────────────
  if (results.frame_details && results.frame_details.length > 0) {
    ensureSpace(40);

    const scores = results.frame_details.map((f) => f.overall_score);
    const minScore = Math.min(...scores);
    const maxScore = Math.max(...scores);
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    const stdDev = Math.sqrt(scores.reduce((sum, s) => sum + (s - avgScore) ** 2, 0) / scores.length);
    const maxFaces = Math.max(...results.frame_details.map((f) => f.faces?.length ?? 0));

    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(...COLORS.accent);
    doc.text("ESTATÍSTICAS DE FRAMES", margin + 6, y + 8);
    y += 14;

    autoTable(doc, {
      startY: y,
      margin: { left: margin, right: margin },
      head: [["Métrica", "Valor"]],
      body: [
        ["Score Mínimo", pct(minScore)],
        ["Score Máximo", pct(maxScore)],
        ["Score Médio", pct(avgScore)],
        ["Desvio Padrão", `${(stdDev * 100).toFixed(2)}%`],
        ["Máx. Faces por Frame", maxFaces.toString()],
        ["Total de Frames", results.frame_details.length.toString()],
      ],
      theme: "plain",
      styles: {
        fillColor: COLORS.card,
        textColor: COLORS.white,
        fontSize: 9,
        cellPadding: 3,
        lineColor: COLORS.border,
        lineWidth: 0.3,
      },
      headStyles: {
        fillColor: [39, 39, 42],
        textColor: COLORS.accent,
        fontStyle: "bold",
      },
      columnStyles: {
        1: { halign: "center", fontStyle: "bold" },
      },
    });

    y = (docWithTable.lastAutoTable?.finalY ?? y) + 6;
  }

  // ── Section: Top suspicious frames (thumbnails) ─────────────────────────
  // Best-effort: requires an apiBase. Frames are downloaded from /api/frame
  // and embedded as JPEGs. This makes the report visually informative without
  // forcing the user to navigate back to the dashboard.
  const thumbs = await fetchTopSuspiciousFrames(data, options.apiBase, options.apiKey, 3);
  if (thumbs.length > 0) {
    ensureSpace(80);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(...COLORS.accent);
    doc.text("FRAMES MAIS SUSPEITOS", margin + 6, y + 8);
    y += 14;

    const thumbW = (contentW - 10) / 3;  // 3 columns, 5mm gap between them
    const thumbH = thumbW * 0.5625;       // 16:9 aspect
    ensureSpace(thumbH + 18);

    for (let i = 0; i < thumbs.length; i++) {
      const t = thumbs[i];
      const x = margin + i * (thumbW + 5);
      try {
        doc.addImage(t.dataUrl, "JPEG", x, y, thumbW, thumbH);
      } catch {
        // jsPDF can throw on malformed data URLs — draw a placeholder box
        doc.setFillColor(...COLORS.card);
        doc.rect(x, y, thumbW, thumbH, "F");
      }
      // Caption under the thumb
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8);
      doc.setTextColor(...COLORS.muted);
      doc.text(
        `Frame ${t.index + 1} · ${t.timestamp.toFixed(1)}s`,
        x, y + thumbH + 5,
      );
      doc.setTextColor(...classifyScore(t.score).rgb);
      doc.setFont("helvetica", "bold");
      doc.text(pct(t.score), x, y + thumbH + 10);
    }
    y += thumbH + 16;
  }

  // ── Section: Technical Notes ────────────────────────────────────────────
  ensureSpace(30);

  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(...COLORS.accent);
  doc.text("NOTAS TÉCNICAS", margin + 6, y + 8);
  y += 14;

  doc.setFillColor(...COLORS.card);
  doc.roundedRect(margin, y, contentW, 22, 2, 2, "F");

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(...COLORS.muted);

  doc.text(`Motor: Deepfake Forensics Engine v2.0 — Sistema de Plugins`, col1, y + 7);
  doc.text(`Modo de análise: ${modeLabel(metadata.analysis_mode)}`, col1, y + 13);
  doc.text(`Plugins ativos: ${metadata.active_plugins}`, col2, y + 7);
  doc.text(`Dashboard: Deepfake Dashboard v2.0`, col2, y + 13);
  doc.text(`Este relatório foi gerado automaticamente. Os scores representam probabilidades estatísticas, não certezas absolutas.`, col1, y + 19);

  // ── Footer ──────────────────────────────────────────────────────────────
  const pageH = doc.internal.pageSize.getHeight();
  doc.setFontSize(7);
  doc.setTextColor(...COLORS.dimmed);
  doc.text(
    `Relatório Forense · ${data.analysis_id} · ${now()} · Página 1`,
    pageW / 2, pageH - 8,
    { align: "center" }
  );

  // ── Save ────────────────────────────────────────────────────────────────
  const safeName = metadata.filename.replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_-]/g, "_");
  doc.save(`relatorio_forense_${safeName}.pdf`);
}
