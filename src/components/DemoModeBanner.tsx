"use client";

import React from "react";
import Link from "next/link";
import { Download, BookOpen, Shield, Cpu } from "lucide-react";
import { useTheme } from "@/hooks/useTheme";

/**
 * Shown on pages that need the local engine to function (the analyser at /
 * and the history viewer at /historico). On the public Vercel deploy those
 * pages have no backend to call, so instead of letting them error out we
 * surface a clear "this is the showcase; download for the real thing"
 * pitch with two CTAs (Download / Learn more).
 *
 * The component receives the page-specific context (which page is hidden,
 * what feature requires the engine) so the copy stays specific rather than
 * a generic "something broke" panel.
 */
interface DemoModeBannerProps {
  /** Short page-specific title, e.g. "Análise de vídeo" or "Histórico". */
  feature: string;
  /** 1-2 sentence explanation of WHY this page needs the local engine. */
  reason: string;
}

export default function DemoModeBanner({ feature, reason }: DemoModeBannerProps) {
  const { isDark } = useTheme();

  return (
    <div
      className={`relative max-w-3xl mx-auto rounded-3xl border overflow-hidden ${
        isDark
          ? "bg-zinc-900/40 border-zinc-800"
          : "bg-white border-zinc-200"
      }`}
    >
      {/* Decorative blur */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute -top-20 -right-20 w-72 h-72 bg-purple-500/10 blur-[80px] rounded-full" />
      </div>

      <div className="relative p-8 md:p-10 space-y-6">
        {/* Badge */}
        <div
          className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold border ${
            isDark
              ? "bg-purple-500/10 border-purple-500/30 text-purple-300"
              : "bg-purple-50 border-purple-300 text-purple-700"
          }`}
        >
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75 animate-ping" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500" />
          </span>
          MODO DEMO
        </div>

        {/* Heading */}
        <div>
          <h2
            className={`text-2xl md:text-3xl font-black tracking-tight mb-2 ${
              isDark ? "text-white" : "text-zinc-900"
            }`}
          >
            {feature} corre na tua máquina
          </h2>
          <p
            className={`text-base ${
              isDark ? "text-zinc-400" : "text-zinc-600"
            }`}
          >
            {reason}
          </p>
        </div>

        {/* Why callout */}
        <div
          className={`flex gap-3 p-4 rounded-xl border ${
            isDark
              ? "bg-purple-500/5 border-purple-500/20"
              : "bg-purple-50/50 border-purple-200"
          }`}
        >
          <Shield className="w-5 h-5 text-purple-400 shrink-0 mt-0.5" />
          <p
            className={`text-sm leading-relaxed ${
              isDark ? "text-zinc-300" : "text-zinc-700"
            }`}
          >
            O motor de análise integra <b>6 detetores ML</b> (~2 GB de modelos
            em memória) e processa cada ficheiro localmente. Não corre em
            free tiers cloud — e essa é a decisão arquitetural: a tua media
            nunca sai da tua máquina.
          </p>
        </div>

        {/* CTAs */}
        <div className="flex flex-col sm:flex-row gap-3 pt-2">
          <Link
            href="/download"
            className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-purple-600 hover:bg-purple-700 text-white font-semibold text-sm transition-colors"
          >
            <Download className="w-4 h-4" />
            Descarregar versão completa
          </Link>
          <Link
            href="/learn"
            className={`inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl border font-semibold text-sm transition-colors ${
              isDark
                ? "border-zinc-700 hover:bg-zinc-800 text-zinc-200"
                : "border-zinc-300 hover:bg-zinc-100 text-zinc-700"
            }`}
          >
            <BookOpen className="w-4 h-4" />
            Como funciona
          </Link>
        </div>

        {/* What you can still do */}
        <div
          className={`pt-4 border-t ${
            isDark ? "border-zinc-800" : "border-zinc-200"
          }`}
        >
          <p
            className={`text-xs uppercase tracking-wider font-semibold mb-3 ${
              isDark ? "text-zinc-500" : "text-zinc-500"
            }`}
          >
            Disponíveis nesta versão pública
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-sm">
            <Link
              href="/learn"
              className={`flex items-center gap-2 p-2 rounded-lg transition-colors ${
                isDark
                  ? "hover:bg-zinc-800 text-zinc-300"
                  : "hover:bg-zinc-100 text-zinc-700"
              }`}
            >
              <BookOpen className="w-4 h-4 text-blue-400" />
              Informação
            </Link>
            <Link
              href="/game"
              className={`flex items-center gap-2 p-2 rounded-lg transition-colors ${
                isDark
                  ? "hover:bg-zinc-800 text-zinc-300"
                  : "hover:bg-zinc-100 text-zinc-700"
              }`}
            >
              <Cpu className="w-4 h-4 text-orange-400" />
              Desafio (real vs fake)
            </Link>
            <Link
              href="/download"
              className={`flex items-center gap-2 p-2 rounded-lg transition-colors ${
                isDark
                  ? "hover:bg-zinc-800 text-zinc-300"
                  : "hover:bg-zinc-100 text-zinc-700"
              }`}
            >
              <Download className="w-4 h-4 text-purple-400" />
              Download
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
