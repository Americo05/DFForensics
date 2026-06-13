"use client";

import React from 'react';
import Link from 'next/link';
import {
  Shield, BookOpen, Cpu, Eye, Info, Brain, Radio, Waves,
  Activity, FileText, Database, Layers,
} from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';
import TopNav from '@/components/TopNav';

// ── Plugin / Analyzer card data ─────────────────────────────────────────────
// Single source of truth so the page mirrors what's actually loaded in the
// engine. When adding a new plugin in engine/plugins/, also extend this list.

interface DetectorCard {
  name: string;
  family: 'visual' | 'audio' | 'metadata' | 'biological' | 'temporal';
  weight?: string;          // headline weight in the dominant scene
  short: string;            // 1-sentence pitch
  long: string;             // 2-3 sentence detail
  reference?: string;       // paper / model
}

const DETECTORS: DetectorCard[] = [
  {
    name: 'MesoNet',
    family: 'visual',
    weight: '0.40 em CROPPED_FACE',
    short: 'CNN compacta (~30k parâmetros) desenhada especificamente para detetar face-swap deepfakes.',
    long: 'Opera na escala mesoscópica — entre micro-texturas e semântica global — onde os algoritmos de troca de rosto deixam as suas assinaturas mais visíveis. É o detetor com maior peso na nossa arquitetura quando a face ocupa grande parte da imagem.',
    reference: 'Afchar et al., WIFS 2018',
  },
  {
    name: 'ViT (Vision Transformer)',
    family: 'visual',
    weight: '0.18 em CROPPED_FACE',
    short: 'Transformer fine-tuned em 140 mil imagens reais e geradas por IA.',
    long: 'Modelo dima806/deepfake_vs_real_image_detection do Hugging Face. Identifica padrões estatísticos globais invisíveis ao olho humano, complementando o foco mesoscópico do MesoNet com uma visão "macro" da imagem.',
    reference: 'Dosovitskiy et al., ICLR 2021',
  },
  {
    name: 'DCT Frequency',
    family: 'visual',
    weight: '0.10 em CROPPED_FACE',
    short: 'Análise da assinatura espectral baseada na lei de potência 1/f² das imagens naturais.',
    long: 'Combina três sinais frequenciais: desvio da lei de potência, picos cruciformes em frequências de stride-2 (típicos de upsampling em GANs), e flatness espectral. Único plugin que funciona em qualquer cena, incluindo paisagens AI sem rosto.',
    reference: 'Zhang et al., WIFS 2019',
  },
  {
    name: 'Edge Blending',
    family: 'visual',
    weight: '0.07 em CROPPED_FACE',
    short: 'Procura artefactos de fusão nas bordas da máscara facial colada.',
    long: 'Analisa gradientes Sobel, descontinuidades de matiz/saturação HSV, e inconsistências de iluminação no anel periférico do rosto detetado. Baseado em Face X-ray; peso baixo após validação empírica que mostrou eficácia limitada em conteúdo moderno re-encodado.',
    reference: 'Li et al., CVPR 2020',
  },
  {
    name: 'PRNU Noise Residue',
    family: 'visual',
    weight: '0.05 em CROPPED_FACE',
    short: 'Compara o "fingerprint" de ruído do sensor entre a face e o resto da cena.',
    long: 'Câmaras reais imprimem padrões de ruído consistentes (PRNU); rostos sintetizados ou colados carregam estatísticas de ruído diferentes do fundo. Implementação simplificada via Gaussian high-pass; peso conservador porque é vulnerável a re-encoding de vídeo.',
    reference: 'Lukáš et al., IEEE TIFS 2006',
  },
  {
    name: 'Sightengine Cloud',
    family: 'visual',
    weight: 'usado em exclusivo no modo Cloud',
    short: 'API comercial cloud para deteção de deepfake e conteúdo AI-generated.',
    long: 'Único plugin externo — opcional, requer credenciais. Quando o utilizador ativa "Cloud API" no upload, este plugin corre sozinho e substitui os detetores locais (não funciona em conjunto com eles). Rate-limited (1 chamada a cada 10 frames) para preservar a quota gratuita.',
  },
];

const ANALYZERS: DetectorCard[] = [
  {
    name: 'Lip Sync (MediaPipe + STFT)',
    family: 'audio',
    short: 'Correlação matemática entre energia do áudio e movimento dos lábios ao longo do tempo.',
    long: 'Usa os 478 landmarks faciais 3D do MediaPipe para medir abertura da boca por frame; cruza com energia espectral do áudio. Em vídeos autênticos, abertura correlaciona-se fortemente com amplitude; em deepfakes com voz sobreposta, esta correlação degrada.',
    reference: 'LipFD, NeurIPS 2024',
  },
  {
    name: 'WavLM Audio Deepfake',
    family: 'audio',
    short: 'Rede neural acústica que distingue voz natural de voz clonada por IA.',
    long: 'Modelo abhishtagatya/wavlm-base-960h-itw-deepfake, treinado em dezenas de horas de áudio real e sintético. Analisa janelas de áudio extraídas via ffmpeg da pista do vídeo.',
    reference: 'Chen et al., IEEE JSTSP 2022',
  },
  {
    name: 'rPPG (Photoplethysmography)',
    family: 'biological',
    short: 'Deteta a pulsação cardíaca observando micro-variações no canal verde do rosto.',
    long: 'A circulação sanguínea causa oscilações imperceptíveis no canal verde (0.7–4 Hz, ou 42–240 bpm). Rostos sintetizados não modelam circulação — ou não têm sinal, ou herdam o ritmo cardíaco da fonte de forma desincronizada.',
    reference: 'FakeCatcher (Ciftci et al., PAMI 2020)',
  },
  {
    name: 'Temporal Coherence',
    family: 'temporal',
    short: 'Verifica se o rosto se move e muda de forma coerente entre frames consecutivos.',
    long: 'Mede instabilidade de landmarks, "saltos" de iluminação, e flickering típicos de pipelines de face-swap aplicados frame a frame. Falhas temporais são uma assinatura clássica de manipulação por GAN.',
  },
  {
    name: 'Metadata Forensics',
    family: 'metadata',
    short: 'Inspeciona EXIF (imagens) e ffprobe (vídeos) à procura de inconsistências.',
    long: 'Examina codecs invulgares, timestamps incoerentes, ausência de assinatura de câmara, perfis de encoding suspeitos. Não detecta a manipulação por si só, mas adiciona contexto valioso ao veredicto final.',
  },
];

// Tailwind v4 only emits classes it finds as full literal strings in the
// source. Dynamic interpolations like `bg-${color}-500/20` are silently
// dropped, so we keep an explicit per-family record of the exact classes
// we want generated.
const FAMILY_STYLE: Record<
  DetectorCard['family'],
  { label: string; icon: React.ComponentType<{ className?: string }>; iconBg: string; iconText: string; chipBg: string; chipText: string }
> = {
  visual:     { label: 'Visual',     icon: Eye,      iconBg: 'bg-purple-500/20', iconText: 'text-purple-400', chipBg: 'bg-purple-500/10', chipText: 'text-purple-400' },
  audio:      { label: 'Áudio',      icon: Waves,    iconBg: 'bg-blue-500/20',   iconText: 'text-blue-400',   chipBg: 'bg-blue-500/10',   chipText: 'text-blue-400'   },
  biological: { label: 'Biológico',  icon: Activity, iconBg: 'bg-red-500/20',    iconText: 'text-red-400',    chipBg: 'bg-red-500/10',    chipText: 'text-red-400'    },
  temporal:   { label: 'Temporal',   icon: Radio,    iconBg: 'bg-orange-500/20', iconText: 'text-orange-400', chipBg: 'bg-orange-500/10', chipText: 'text-orange-400' },
  metadata:   { label: 'Metadata',   icon: FileText, iconBg: 'bg-green-500/20',  iconText: 'text-green-400',  chipBg: 'bg-green-500/10',  chipText: 'text-green-400'  },
};

function DetectorList({
  items,
  isDark,
}: { items: DetectorCard[]; isDark: boolean }) {
  return (
    <div className="space-y-3">
      {items.map((d) => {
        const style = FAMILY_STYLE[d.family];
        const FamilyIcon = style.icon;
        return (
          <article
            key={d.name}
            className={`rounded-xl border p-4 ${
              isDark
                ? 'bg-zinc-900/40 border-zinc-800'
                : 'bg-white border-zinc-200'
            }`}
          >
            <div className="flex items-start justify-between gap-3 mb-2">
              <div className="flex items-center gap-3 min-w-0">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${style.iconBg}`}>
                  <FamilyIcon className={`w-5 h-5 ${style.iconText}`} />
                </div>
                <div className="min-w-0">
                  <h3 className={`font-bold truncate ${isDark ? 'text-white' : 'text-zinc-900'}`}>
                    {d.name}
                  </h3>
                  <div className="flex flex-wrap items-center gap-2 mt-0.5">
                    <span className={`text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded ${style.chipBg} ${style.chipText}`}>
                      {style.label}
                    </span>
                    {d.weight && (
                      <span className={`text-[10px] ${isDark ? 'text-zinc-500' : 'text-zinc-500'}`}>
                        peso {d.weight}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>
            <p className={`text-sm font-medium mb-1 ${isDark ? 'text-zinc-200' : 'text-zinc-800'}`}>
              {d.short}
            </p>
            <p className={`text-xs leading-relaxed ${isDark ? 'text-zinc-400' : 'text-zinc-600'}`}>
              {d.long}
            </p>
            {d.reference && (
              <p className={`mt-2 text-[10px] italic ${isDark ? 'text-zinc-500' : 'text-zinc-500'}`}>
                Referência: {d.reference}
              </p>
            )}
          </article>
        );
      })}
    </div>
  );
}

export default function LearnPage() {
  const { isDark } = useTheme();

  return (
    <main className={`min-h-screen selection:bg-purple-500/30 transition-colors duration-300 ${
      isDark ? 'bg-[#050505] text-white' : 'bg-[#f8f9fb] text-zinc-900'
    }`}>
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-purple-600/20 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-600/10 blur-[120px] rounded-full" />
      </div>

      <TopNav />

      <div className="relative z-10 max-w-4xl mx-auto px-6 pt-32 pb-20">

        {/* ── Hero ───────────────────────────────────────────────── */}
        <div className="text-center mb-16">
          <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full glass text-sm mb-6 border ${
            isDark ? 'text-zinc-300 border-zinc-800' : 'text-zinc-600 border-zinc-300'
          }`}>
            <BookOpen className="w-4 h-4 text-purple-400" />
            <span>Guia Educativo</span>
          </div>
          <h1
            className="text-4xl md:text-6xl font-black tracking-tight mb-6 bg-gradient-to-br from-white via-white to-zinc-500 bg-clip-text text-transparent"
            style={!isDark ? { backgroundImage: 'linear-gradient(to bottom right, #18181b, #18181b, #71717a)' } : undefined}
          >
            Compreender os <br/> Deepfakes
          </h1>
          <p className={`text-lg font-light max-w-2xl mx-auto ${isDark ? 'text-zinc-400' : 'text-zinc-600'}`}>
            Como a desinformação digital é criada — e como a nossa arquitetura forense, baseada em <b>6 detetores</b> e <b>5 análises vídeo-level</b>, a consegue detetar.
          </p>
        </div>

        <div className="space-y-12">

          {/* ── Section 1: O problema ─────────────────────────────── */}
          <section className={`glass p-8 rounded-3xl border ${isDark ? 'border-zinc-800' : 'border-zinc-200'}`}>
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center">
                <Info className="w-6 h-6 text-purple-400" />
              </div>
              <h2 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-zinc-900'}`}>O que são Deepfakes?</h2>
            </div>
            <p className={`leading-relaxed ${isDark ? 'text-zinc-300' : 'text-zinc-600'}`}>
              Os <b>Deepfakes</b> são conteúdos sintéticos hiper-realistas gerados por <b>Redes Adversárias Generativas (GANs)</b> e modelos de difusão latente. Estas manipulações desafiam a perceção humana — que opera próximo do acaso (~50%) na deteção visual — e representam ameaças reais à segurança, integridade democrática e privacidade individual.
            </p>
          </section>

          {/* ── Section 2: Como são criados ───────────────────────── */}
          <section className={`glass p-8 rounded-3xl border ${isDark ? 'border-zinc-800' : 'border-zinc-200'}`}>
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center">
                <Cpu className="w-6 h-6 text-blue-400" />
              </div>
              <h2 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-zinc-900'}`}>Como são criados?</h2>
            </div>
            <p className={`leading-relaxed ${isDark ? 'text-zinc-300' : 'text-zinc-600'}`}>
              Pipelines automatizados treinam modelos sobre as características faciais para depois sobrepor e recriar o rosto (face swap). Artefactos clássicos: <i>&ldquo;blending&rdquo;</i> nas bordas das máscaras, discrepâncias de iluminação, dessincronização entre lábios e voz, e ausência de sinais biológicos como pulsação cardíaca visível na pele.
            </p>
          </section>

          {/* ── Section 3: Arquitetura — os 6 detetores ───────────── */}
          <section className={`glass p-8 rounded-3xl border ${isDark ? 'border-zinc-800' : 'border-zinc-200'}`}>
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 rounded-xl bg-green-500/20 flex items-center justify-center">
                <Layers className="w-6 h-6 text-green-400" />
              </div>
              <div>
                <h2 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-zinc-900'}`}>
                  Os 6 detetores por-frame
                </h2>
                <p className={`text-xs mt-1 ${isDark ? 'text-zinc-500' : 'text-zinc-500'}`}>
                  Cada plugin analisa cada frame individualmente. Um <i>SceneClassifier</i> decide quais correm consoante o tipo de cena (face próxima, face em contexto, ou sem face).
                </p>
              </div>
            </div>
            <DetectorList items={DETECTORS} isDark={isDark} />
          </section>

          {/* ── Section 4: Analyzers video-level ──────────────────── */}
          <section className={`glass p-8 rounded-3xl border ${isDark ? 'border-zinc-800' : 'border-zinc-200'}`}>
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center">
                <Brain className="w-6 h-6 text-blue-400" />
              </div>
              <div>
                <h2 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-zinc-900'}`}>
                  5 análises vídeo-level
                </h2>
                <p className={`text-xs mt-1 ${isDark ? 'text-zinc-500' : 'text-zinc-500'}`}>
                  Operam sobre o vídeo completo (não frame-a-frame), explorando sinais temporais, áudio e biológicos.
                </p>
              </div>
            </div>
            <DetectorList items={ANALYZERS} isDark={isDark} />
          </section>

          {/* ── Section 5: Histórico local ────────────────────────── */}
          <section className={`glass p-8 rounded-3xl border ${isDark ? 'border-zinc-800' : 'border-zinc-200'}`}>
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center">
                <Database className="w-6 h-6 text-purple-400" />
              </div>
              <h2 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-zinc-900'}`}>
                Histórico 100% local
              </h2>
            </div>
            <p className={`leading-relaxed ${isDark ? 'text-zinc-300' : 'text-zinc-600'}`}>
              Todas as análises ficam guardadas numa base SQLite em <code className={isDark ? 'text-purple-400' : 'text-purple-600'}>~/.deepfake-forensics/history.db</code> na máquina do utilizador. Nenhum dado de upload sai do computador. A página <Link href="/historico" className={isDark ? 'text-purple-400 hover:text-purple-300 underline' : 'text-purple-600 hover:text-purple-700 underline'}>Histórico</Link> permite rever, comparar e eliminar análises anteriores.
            </p>
          </section>

          {/* ── Section 6: Identificar a olho nu ──────────────────── */}
          <section className={`glass p-8 rounded-3xl border ${isDark ? 'border-zinc-800' : 'border-zinc-200'}`}>
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 rounded-xl bg-yellow-500/20 flex items-center justify-center">
                <Eye className="w-6 h-6 text-yellow-400" />
              </div>
              <h2 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-zinc-900'}`}>
                Como identificar a olho nu?
              </h2>
            </div>
            <p className={`leading-relaxed ${isDark ? 'text-zinc-300' : 'text-zinc-600'}`}>
              Mesmo sem ferramentas, há pistas úteis: piscar de olhos pouco natural, texturas esborratadas nas fronteiras do rosto causadas por falhas de <i>&ldquo;blending&rdquo;</i>, e dessincronização entre palavras e movimento dos lábios. A perceção humana continua a ser uma boa primeira linha de defesa, especialmente quando complementada pela forense digital.
            </p>
            <div className={`mt-4 p-3 rounded-lg border ${
              isDark ? 'bg-purple-500/5 border-purple-500/30' : 'bg-purple-50 border-purple-200'
            }`}>
              <p className={`text-sm flex items-start gap-2 ${isDark ? 'text-zinc-300' : 'text-zinc-700'}`}>
                <Shield className="w-4 h-4 text-purple-400 shrink-0 mt-0.5" />
                <span>
                  Treina o teu olho no <Link href="/game" className={isDark ? 'text-purple-400 hover:text-purple-300 underline' : 'text-purple-600 hover:text-purple-700 underline'}>Desafio</Link> — distingue rostos reais de rostos gerados por IA num jogo curto.
                </span>
              </p>
            </div>
          </section>

        </div>
      </div>
    </main>
  );
}
