"use client";

import React, { useState } from 'react';
import {
  Download,
  Github,
  Check,
  Copy,
  Terminal,
  HardDrive,
  Cpu,
  Shield,
  Globe,
  Box,
  AlertTriangle,
  ExternalLink,
} from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';
import TopNav from '@/components/TopNav';

const GITHUB_REPO_URL = 'https://github.com/Americo05/DFForensics';
const GITHUB_RELEASE_URL = `${GITHUB_REPO_URL}/releases/latest`;
const GITHUB_ZIP_URL = `${GITHUB_REPO_URL}/archive/refs/heads/main.zip`;
const DOCKER_DESKTOP_URL = 'https://www.docker.com/products/docker-desktop/';

const CLONE_CMD = `git clone ${GITHUB_REPO_URL}.git`;
const RUN_CMD = 'docker compose up --build';

// ── Reusable: copy-to-clipboard code block ───────────────────────────────
function CodeBlock({ code, isDark }: { code: string; isDark: boolean }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard API can fail on http:// or in some embedded contexts —
      // user can still select manually
    }
  };

  return (
    <div
      className={`flex items-center justify-between gap-3 rounded-lg border px-4 py-3 font-mono text-sm ${
        isDark
          ? 'bg-zinc-950 border-zinc-800 text-zinc-200'
          : 'bg-zinc-100 border-zinc-300 text-zinc-800'
      }`}
    >
      <code className="truncate select-all">{code}</code>
      <button
        onClick={handleCopy}
        className={`shrink-0 inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-sans transition-colors ${
          isDark
            ? 'bg-zinc-800 hover:bg-zinc-700 text-zinc-200'
            : 'bg-zinc-200 hover:bg-zinc-300 text-zinc-800'
        }`}
        title="Copiar para a área de transferência"
      >
        {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
        {copied ? 'Copiado' : 'Copiar'}
      </button>
    </div>
  );
}

// ── Reusable: requirement chip ───────────────────────────────────────────
function Requirement({
  icon,
  label,
  isDark,
}: {
  icon: React.ReactNode;
  label: string;
  isDark: boolean;
}) {
  return (
    <div
      className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm ${
        isDark
          ? 'bg-zinc-900/50 border-zinc-800 text-zinc-300'
          : 'bg-white border-zinc-200 text-zinc-700'
      }`}
    >
      {icon}
      <span>{label}</span>
    </div>
  );
}

// ── Comparison table row ─────────────────────────────────────────────────
function CompareRow({
  feature,
  web,
  local,
  isDark,
}: {
  feature: string;
  web: React.ReactNode;
  local: React.ReactNode;
  isDark: boolean;
}) {
  return (
    <tr
      className={`border-b last:border-b-0 ${
        isDark ? 'border-zinc-800' : 'border-zinc-200'
      }`}
    >
      <td className="py-3 px-4 text-sm">{feature}</td>
      <td className="py-3 px-4 text-sm text-center">{web}</td>
      <td className="py-3 px-4 text-sm text-center font-semibold text-purple-400">
        {local}
      </td>
    </tr>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────
export default function DownloadPage() {
  const { isDark } = useTheme();

  return (
    <main
      className={`min-h-screen selection:bg-purple-500/30 transition-colors duration-300 ${
        isDark ? 'bg-[#050505] text-white' : 'bg-[#f8f9fb] text-zinc-900'
      }`}
    >
      {/* Background blurs to match other pages */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-purple-600/20 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-600/10 blur-[120px] rounded-full" />
      </div>

      <TopNav />

      <div className="relative z-10 max-w-4xl mx-auto px-6 pt-32 pb-20">
        {/* ── Header ─────────────────────────────────────────────────── */}
        <div className="text-center mb-12">
          <div
            className={`inline-flex items-center gap-2 px-3 py-1 rounded-full glass text-sm mb-6 border ${
              isDark ? 'text-zinc-300 border-zinc-800' : 'text-zinc-600 border-zinc-300'
            }`}
          >
            <Download className="w-4 h-4 text-purple-400" />
            <span>Versão Completa Local</span>
          </div>
          <h1
            className="text-4xl md:text-6xl font-black tracking-tight mb-6 bg-gradient-to-br from-white via-white to-zinc-500 bg-clip-text text-transparent"
            style={
              !isDark
                ? { backgroundImage: 'linear-gradient(to bottom right, #18181b, #18181b, #71717a)' }
                : undefined
            }
          >
            Instala e corre <br /> na tua máquina
          </h1>
          <p
            className={`text-lg font-light max-w-2xl mx-auto ${
              isDark ? 'text-zinc-400' : 'text-zinc-600'
            }`}
          >
            Sem limites de análises, sem ficheiros enviados para a cloud. Toda a deteção
            corre no teu hardware, usando os 6 detetores em conjunto.
          </p>
        </div>

        {/* ── Comparison ─────────────────────────────────────────────── */}
        <section
          className={`glass p-6 rounded-2xl border mb-12 ${
            isDark ? 'border-zinc-800' : 'border-zinc-200'
          }`}
        >
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Globe className="w-5 h-5 text-blue-400" />
            Versão Web vs Versão Local
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr
                  className={`border-b ${
                    isDark ? 'border-zinc-800 text-zinc-400' : 'border-zinc-300 text-zinc-600'
                  }`}
                >
                  <th className="text-left py-2 px-4 font-medium text-sm">Funcionalidade</th>
                  <th className="text-center py-2 px-4 font-medium text-sm">Web (demo)</th>
                  <th className="text-center py-2 px-4 font-medium text-sm">Local (completo)</th>
                </tr>
              </thead>
              <tbody>
                <CompareRow
                  feature="Análises por dia"
                  web="3"
                  local="∞"
                  isDark={isDark}
                />
                <CompareRow
                  feature="Tamanho máximo do ficheiro"
                  web="20 MB"
                  local="200 MB"
                  isDark={isDark}
                />
                <CompareRow
                  feature="Detetores ativos"
                  web="ViT + DCT"
                  local="Os 6 todos"
                  isDark={isDark}
                />
                <CompareRow
                  feature="Análise áudio (lip-sync + WavLM)"
                  web="—"
                  local="✓"
                  isDark={isDark}
                />
                <CompareRow
                  feature="Histórico persistente"
                  web="—"
                  local="✓ (SQLite local)"
                  isDark={isDark}
                />
                <CompareRow
                  feature="Privacidade dos uploads"
                  web="Servidor partilhado"
                  local="100% na tua máquina"
                  isDark={isDark}
                />
              </tbody>
            </table>
          </div>
        </section>

        {/* ── Requirements ───────────────────────────────────────────── */}
        <section
          className={`glass p-6 rounded-2xl border mb-12 ${
            isDark ? 'border-zinc-800' : 'border-zinc-200'
          }`}
        >
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-purple-400" />
            Requisitos
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
            <Requirement
              icon={<HardDrive className="w-4 h-4 text-purple-400" />}
              label="5 GB de disco"
              isDark={isDark}
            />
            <Requirement
              icon={<Cpu className="w-4 h-4 text-purple-400" />}
              label="4 GB RAM (8 GB ideal)"
              isDark={isDark}
            />
            <Requirement
              icon={<Box className="w-4 h-4 text-purple-400" />}
              label="Docker Desktop"
              isDark={isDark}
            />
          </div>
          <a
            href={DOCKER_DESKTOP_URL}
            target="_blank"
            rel="noopener noreferrer"
            className={`inline-flex items-center gap-2 text-sm transition-colors ${
              isDark ? 'text-blue-400 hover:text-blue-300' : 'text-blue-600 hover:text-blue-700'
            }`}
          >
            <ExternalLink className="w-4 h-4" />
            Instalar Docker Desktop (Windows / macOS / Linux)
          </a>
        </section>

        {/* ── Steps ──────────────────────────────────────────────────── */}
        <section
          className={`glass p-6 rounded-2xl border mb-12 ${
            isDark ? 'border-zinc-800' : 'border-zinc-200'
          }`}
        >
          <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
            <Terminal className="w-5 h-5 text-purple-400" />
            Instalação em 3 passos
          </h2>

          {/* Step 1 */}
          <div className="mb-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-7 h-7 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center text-sm font-bold">
                1
              </div>
              <h3 className="font-semibold">Descarrega o código</h3>
            </div>
            <p
              className={`text-sm mb-3 ml-10 ${
                isDark ? 'text-zinc-400' : 'text-zinc-600'
              }`}
            >
              Por <code className={isDark ? 'text-purple-400' : 'text-purple-600'}>git</code> (recomendado, permite atualizar) ou ZIP direto.
            </p>
            <div className="ml-10 space-y-3">
              <CodeBlock code={CLONE_CMD} isDark={isDark} />
              <div className="flex flex-wrap gap-3">
                <a
                  href={GITHUB_REPO_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-colors ${
                    isDark
                      ? 'border-zinc-700 hover:bg-zinc-900 text-zinc-200'
                      : 'border-zinc-300 hover:bg-zinc-100 text-zinc-800'
                  }`}
                >
                  <Github className="w-4 h-4" />
                  Repositório GitHub
                </a>
                <a
                  href={GITHUB_ZIP_URL}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium transition-colors"
                >
                  <Download className="w-4 h-4" />
                  Download ZIP (main)
                </a>
                <a
                  href={GITHUB_RELEASE_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-colors ${
                    isDark
                      ? 'border-zinc-700 hover:bg-zinc-900 text-zinc-200'
                      : 'border-zinc-300 hover:bg-zinc-100 text-zinc-800'
                  }`}
                >
                  <ExternalLink className="w-4 h-4" />
                  Última Release
                </a>
              </div>
            </div>
          </div>

          {/* Step 2 */}
          <div className="mb-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-7 h-7 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center text-sm font-bold">
                2
              </div>
              <h3 className="font-semibold">Levanta a aplicação</h3>
            </div>
            <p
              className={`text-sm mb-3 ml-10 ${
                isDark ? 'text-zinc-400' : 'text-zinc-600'
              }`}
            >
              Abre um terminal na pasta <code className={isDark ? 'text-purple-400' : 'text-purple-600'}>DFForensics</code> e corre:
            </p>
            <div className="ml-10">
              <CodeBlock code={RUN_CMD} isDark={isDark} />
              <p
                className={`text-xs mt-2 ${
                  isDark ? 'text-zinc-500' : 'text-zinc-500'
                }`}
              >
                A primeira build demora ~5–10 minutos (descarrega PyTorch, MesoNet e o ViT).
                As execuções seguintes arrancam em ~30 segundos.
              </p>
            </div>
          </div>

          {/* Step 3 */}
          <div>
            <div className="flex items-center gap-3 mb-3">
              <div className="w-7 h-7 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center text-sm font-bold">
                3
              </div>
              <h3 className="font-semibold">Abre no browser</h3>
            </div>
            <div className="ml-10">
              <CodeBlock code="http://localhost:3000" isDark={isDark} />
            </div>
          </div>
        </section>

        {/* ── Privacy callout ────────────────────────────────────────── */}
        <section
          className={`p-6 rounded-2xl border mb-12 ${
            isDark
              ? 'bg-purple-500/5 border-purple-500/30'
              : 'bg-purple-50 border-purple-200'
          }`}
        >
          <div className="flex gap-4">
            <Shield className="w-6 h-6 text-purple-400 shrink-0" />
            <div>
              <h3 className="font-semibold mb-2">Porquê correr localmente</h3>
              <p
                className={`text-sm leading-relaxed ${
                  isDark ? 'text-zinc-300' : 'text-zinc-700'
                }`}
              >
                Material sensível — declarações políticas, evidência forense, conteúdo
                não-consensual — não deve ser carregado para servidores de terceiros para
                ser analisado. A versão local garante que o ficheiro nunca sai do teu
                computador. Os scores e o histórico ficam armazenados numa base SQLite
                local em <code className={isDark ? 'text-purple-400' : 'text-purple-600'}>~/.deepfake-forensics/</code>.
              </p>
            </div>
          </div>
        </section>

        {/* ── Troubleshooting ────────────────────────────────────────── */}
        <section
          className={`glass p-6 rounded-2xl border ${
            isDark ? 'border-zinc-800' : 'border-zinc-200'
          }`}
        >
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-yellow-400" />
            Problemas comuns
          </h2>
          <dl className={`space-y-4 text-sm ${isDark ? 'text-zinc-300' : 'text-zinc-700'}`}>
            <div>
              <dt className="font-semibold mb-1">
                <code className={isDark ? 'text-yellow-300' : 'text-yellow-700'}>
                  &quot;docker: command not found&quot;
                </code>
              </dt>
              <dd className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>
                Docker Desktop não está instalado ou não está em execução. Abre a aplicação
                Docker Desktop e espera pelo ícone ficar estável antes de tentar novamente.
              </dd>
            </div>
            <div>
              <dt className="font-semibold mb-1">
                <code className={isDark ? 'text-yellow-300' : 'text-yellow-700'}>
                  Port 3000 (ou 8000) already in use
                </code>
              </dt>
              <dd className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>
                Outra aplicação está a usar essas portas. Fecha-a, ou edita o
                <code className="px-1"> docker-compose.yml </code>
                para mapear portas livres (ex:
                <code className="px-1"> 3001:3000 </code>).
              </dd>
            </div>
            <div>
              <dt className="font-semibold mb-1">
                Análise demora muito tempo
              </dt>
              <dd className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>
                A primeira análise carrega modelos para memória (~30s). Análises seguintes
                são muito mais rápidas. Para vídeos longos, espera entre 30s e 2min em
                CPU típica.
              </dd>
            </div>
          </dl>
          <a
            href={`${GITHUB_REPO_URL}/blob/main/DEPLOYMENT.md`}
            target="_blank"
            rel="noopener noreferrer"
            className={`inline-flex items-center gap-2 mt-4 text-sm transition-colors ${
              isDark ? 'text-blue-400 hover:text-blue-300' : 'text-blue-600 hover:text-blue-700'
            }`}
          >
            <ExternalLink className="w-4 h-4" />
            Guia completo de deployment (DEPLOYMENT.md)
          </a>
        </section>
      </div>
    </main>
  );
}
