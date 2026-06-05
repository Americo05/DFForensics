"use client";

import React, { useCallback, useEffect, useState } from 'react';
import { Heart, HeartCrack, Trophy, Flame, CheckCircle, XCircle, Info } from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';
import TopNav from '@/components/TopNav';

// ── Dataset model ─────────────────────────────────────────────────────────
// `source` lets the player see where the image is coming from (built-in
// fallback vs the user's local sample). Kept purely informational.

type GameImage = {
  id: string;
  url: string;
  isFake: boolean;
  source: 'local' | 'fallback';
};

interface Manifest {
  version: number;
  real: string[];
  fake: string[];
  total: number;
}

// ── Built-in fallback set ─────────────────────────────────────────────────
// Used when public/game-images/manifest.json is absent (e.g. on the Vercel
// public demo). Curated stable Unsplash portraits for real; five sample
// StyleGAN-generated faces shipped under public/game-images/fallback-fakes/
// (no real person, so safe to redistribute).

const FALLBACK_REAL_URLS: string[] = [
  // The four URLs that shipped with the original /game implementation.
  'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=500&q=80',
  'https://images.unsplash.com/photo-1554151228-14d9def656e4?w=500&q=80',
  'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=500&q=80',
  'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=500&q=80',
  // Six additional portraits picked from the Unsplash "faces" search.
  'https://plus.unsplash.com/premium_photo-1671656349322-41de944d259b?w=500&q=80',
  'https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?w=500&q=80',
  'https://plus.unsplash.com/premium_photo-1664203068007-52240d0ca48f?w=500&q=80',
  'https://images.unsplash.com/photo-1542909168-82c3e7fdca5c?w=500&q=80',
  'https://images.unsplash.com/photo-1499996860823-5214fcc65f8f?w=500&q=80',
  'https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=500&q=80',
];

const FALLBACK_FAKE_URLS: string[] = [
  '/game-images/fallback-fakes/fake_01.jpg',
  '/game-images/fallback-fakes/fake_02.jpg',
  '/game-images/fallback-fakes/fake_03.jpg',
  '/game-images/fallback-fakes/fake_04.jpg',
  '/game-images/fallback-fakes/fake_05.jpg',
];

function buildFallbackPool(): GameImage[] {
  const real = FALLBACK_REAL_URLS.map((url, i) => ({
    id: `fb-real-${i}`,
    url,
    isFake: false,
    source: 'fallback' as const,
  }));
  const fake = FALLBACK_FAKE_URLS.map((url, i) => ({
    id: `fb-fake-${i}`,
    url,
    isFake: true,
    source: 'fallback' as const,
  }));
  return [...real, ...fake];
}

function manifestToPool(m: Manifest): GameImage[] {
  const real = m.real.map((url, i) => ({
    id: `local-real-${i}`,
    url,
    isFake: false,
    source: 'local' as const,
  }));
  const fake = m.fake.map((url, i) => ({
    id: `local-fake-${i}`,
    url,
    isFake: true,
    source: 'local' as const,
  }));
  return [...real, ...fake];
}

// Random pick that avoids the most recent image so the player doesn't see
// the same face twice in a row even with a small fallback pool.
function pickNext(pool: GameImage[], previousId: string | null): GameImage {
  if (pool.length === 1) return pool[0];
  let candidate = pool[Math.floor(Math.random() * pool.length)];
  while (candidate.id === previousId) {
    candidate = pool[Math.floor(Math.random() * pool.length)];
  }
  return candidate;
}

// ── localStorage helper for the score persistence ─────────────────────────

const readInt = (key: string, fallback: number) => {
  if (typeof window === 'undefined') return fallback;
  const v = localStorage.getItem(key);
  return v ? parseInt(v) : fallback;
};

// ── Page ──────────────────────────────────────────────────────────────────

export default function GamePage() {
  const { isDark } = useTheme();

  // Score state — lazy-initialised from localStorage to avoid a useEffect
  // that would trip react-hooks/set-state-in-effect.
  const [streak, setStreak] = useState(() => readInt('df_streak', 0));
  const [highScore, setHighScore] = useState(() => readInt('df_highscore', 0));
  const [lives, setLives] = useState(() => readInt('df_lives', 3));

  // Dataset state — pool starts on the fallback, and the manifest fetch
  // (below) may upgrade it. currentImage is seeded with the same fallback
  // so we never render an empty stage; updating it atomically with the
  // pool inside the fetch handler avoids a "setState in useEffect" warning.
  const [pool, setPool] = useState<GameImage[]>(() => buildFallbackPool());
  const [poolSource, setPoolSource] = useState<'local' | 'fallback'>('fallback');
  const [currentImage, setCurrentImage] = useState<GameImage | null>(() => {
    const initial = buildFallbackPool();
    return initial[Math.floor(Math.random() * initial.length)] ?? null;
  });
  const [feedback, setFeedback] = useState<'correct' | 'wrong' | null>(null);

  // Try to upgrade the pool to the local manifest. Silently keeps the
  // fallback if the file isn't present (Vercel deploys without populated
  // local images, which is the expected demo flow).
  useEffect(() => {
    let cancelled = false;
    fetch('/game-images/manifest.json', { cache: 'no-cache' })
      .then((res) => (res.ok ? res.json() : null))
      .then((data: Manifest | null) => {
        if (cancelled || !data || !data.real?.length || !data.fake?.length) return;
        const localPool = manifestToPool(data);
        setPool(localPool);
        setPoolSource('local');
        // Pick a fresh image from the new pool right here so we don't
        // need a separate "react to pool change" effect.
        setCurrentImage(pickNext(localPool, null));
      })
      .catch(() => {
        /* manifest missing → keep fallback */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Persist scores. Same pattern as before; condensed into one effect.
  useEffect(() => {
    localStorage.setItem('df_highscore', highScore.toString());
    localStorage.setItem('df_streak', streak.toString());
    localStorage.setItem('df_lives', lives.toString());
  }, [highScore, streak, lives]);

  const advance = useCallback(() => {
    setFeedback(null);
    setCurrentImage((prev) => pickNext(pool, prev?.id ?? null));
  }, [pool]);

  const handleGuess = (userGuessedFake: boolean) => {
    if (!currentImage || lives === 0 || feedback !== null) return;

    const isCorrect = userGuessedFake === currentImage.isFake;

    if (isCorrect) {
      setFeedback('correct');
      const newStreak = streak + 1;
      setStreak(newStreak);
      if (newStreak > highScore) setHighScore(newStreak);
    } else {
      setFeedback('wrong');
      const newLives = lives - 1;
      setLives(newLives);

      if (newLives === 0) {
        // Game Over — reset after a beat so the player reads the feedback.
        setTimeout(() => {
          alert(`Game Over! A tua streak de ${streak} acabou.`);
          setStreak(0);
          setLives(3);
        }, 1000);
      }
    }

    // Move on after the feedback animation.
    setTimeout(advance, 1200);
  };

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

        {/* Score bar */}
        <div className={`glass p-4 rounded-2xl border flex justify-between items-center mb-4 ${
          isDark ? 'border-zinc-800' : 'border-zinc-200'
        }`}>
          <div className="flex items-center gap-3">
            <Trophy className="w-5 h-5 text-yellow-400" />
            <span className={`font-bold ${isDark ? 'text-white' : 'text-zinc-900'}`}>Best: {highScore}</span>
          </div>

          <div className="flex items-center gap-2">
            {[1, 2, 3].map((life) => (
              <span key={life}>
                {life <= lives ? (
                  <Heart className="w-6 h-6 text-red-500 fill-red-500 drop-shadow-[0_0_8px_rgba(239,68,68,0.5)]" />
                ) : (
                  <HeartCrack className="w-6 h-6 text-zinc-600" />
                )}
              </span>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <Flame className={`w-5 h-5 ${streak > 2 ? 'text-orange-500 fill-orange-500 animate-pulse' : 'text-zinc-400'}`} />
            <span className={`font-bold ${isDark ? 'text-white' : 'text-zinc-900'}`}>Streak: {streak}</span>
          </div>
        </div>

        {/* Pool-source hint — visible but unobtrusive */}
        <div className={`mb-6 flex items-center justify-center gap-2 text-xs ${
          isDark ? 'text-zinc-500' : 'text-zinc-500'
        }`}>
          <Info className="w-3.5 h-3.5" />
          {poolSource === 'local' ? (
            <span>
              A jogar com o teu dataset local ({pool.length} imagens).
            </span>
          ) : (
            <span>
              Modo demo ({pool.length} imagens).
              Instala localmente e corre <code className={isDark ? 'text-purple-400' : 'text-purple-600'}>build_game_dataset.py</code> para mais variedade.
            </span>
          )}
        </div>

        {/* Image stage */}
        <div className="flex flex-col items-center">
          <div className={`relative w-full max-w-lg aspect-square rounded-3xl overflow-hidden glass border-4 transition-colors duration-300
            ${feedback === 'correct' ? 'border-green-500 shadow-[0_0_40px_-10px_rgba(34,197,94,0.5)]' :
              feedback === 'wrong' ? 'border-red-500 shadow-[0_0_40px_-10px_rgba(239,68,68,0.5)]' :
              isDark ? 'border-zinc-800' : 'border-zinc-200'}
          `}>

            {feedback && (
              <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/40 backdrop-blur-sm">
                {feedback === 'correct' ? (
                  <div className="text-center animate-bounce">
                    <CheckCircle className="w-24 h-24 text-green-400 mx-auto mb-2" />
                    <span className="text-2xl font-black text-green-400 drop-shadow-md">CORRETO!</span>
                  </div>
                ) : (
                  <div className="text-center">
                    <XCircle className="w-24 h-24 text-red-400 mx-auto mb-2" />
                    <span className="text-2xl font-black text-red-400 drop-shadow-md">FALHOU!</span>
                    <p className="text-white mt-2 font-bold text-lg">
                      Isto era {currentImage?.isFake ? 'um Deepfake' : 'Real'}
                    </p>
                  </div>
                )}
              </div>
            )}

            {currentImage ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={currentImage.url}
                alt="Real ou Fake?"
                className="object-cover w-full h-full pointer-events-none select-none"
              />
            ) : (
              <div className="flex items-center justify-center w-full h-full">
                <span className="text-zinc-500 animate-pulse">A carregar desafio...</span>
              </div>
            )}
          </div>

          {/* Action buttons */}
          <div className="flex gap-6 mt-10 w-full max-w-lg">
            <button
              onClick={() => handleGuess(false)}
              disabled={feedback !== null || lives === 0}
              className={`flex-1 py-5 rounded-2xl font-black text-xl tracking-wide transition-all
                ${isDark ? 'bg-zinc-800 text-white hover:bg-green-600/90 hover:border-green-500' : 'bg-white text-zinc-900 border hover:bg-green-500 hover:text-white'}
                ${feedback ? 'opacity-50 cursor-not-allowed' : 'active:scale-95 shadow-xl'}
              `}
            >
              REAL
            </button>
            <button
              onClick={() => handleGuess(true)}
              disabled={feedback !== null || lives === 0}
              className={`flex-1 py-5 rounded-2xl font-black text-xl tracking-wide transition-all
                ${isDark ? 'bg-zinc-800 text-white hover:bg-red-600/90 hover:border-red-500' : 'bg-white text-zinc-900 border hover:bg-red-500 hover:text-white'}
                ${feedback ? 'opacity-50 cursor-not-allowed' : 'active:scale-95 shadow-xl'}
              `}
            >
              FAKE
            </button>
          </div>

        </div>
      </div>
    </main>
  );
}
