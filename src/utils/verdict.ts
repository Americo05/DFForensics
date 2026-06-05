/**
 * verdict.ts — Forensic verdict categorization.
 *
 * Why this isn't just `score > 0.6`:
 *   A binary "MANIPULATION DETECTED" / "AUTHENTIC" label gives the same weight
 *   to a borderline 0.61 score and a near-certain 0.99 score. That's misleading
 *   in a forensic context where uncertainty should be communicated, not hidden.
 *
 * Categories follow common practice in forensic reports:
 *   - clean:        score < 0.40   → "Sem evidência de manipulação"
 *   - inconclusive: 0.40–0.60      → "Inconclusivo (recomenda-se análise adicional)"
 *   - suspect:      0.60–0.85      → "Consistente com manipulação"
 *   - strong:       ≥ 0.85         → "Manipulação altamente provável"
 *
 * If you change the thresholds, update them HERE — both the dashboard and
 * the PDF report import from this module.
 */

export type VerdictLevel = 'clean' | 'inconclusive' | 'suspect' | 'strong';

export interface Verdict {
  level: VerdictLevel;
  /** Short uppercase label for badges/cards. */
  label: string;
  /** Longer phrasing for cards/PDF. */
  description: string;
  /** Tailwind color name segment (e.g. 'red', 'green', 'yellow', 'orange'). */
  colorKey: 'green' | 'yellow' | 'orange' | 'red';
  /** RGB tuple for jsPDF (which takes individual numbers). */
  rgb: [number, number, number];
  /** True for suspect/strong — convenient for boolean styling. */
  isFake: boolean;
}

const T_INCONCLUSIVE = 0.40;
const T_SUSPECT      = 0.60;
const T_STRONG       = 0.85;

const VERDICTS: Record<VerdictLevel, Omit<Verdict, 'level' | 'isFake'>> = {
  clean: {
    label: 'SEM EVIDÊNCIA',
    description: 'Sem evidência de manipulação',
    colorKey: 'green',
    rgb: [34, 197, 94],
  },
  inconclusive: {
    label: 'INCONCLUSIVO',
    description: 'Inconclusivo — recomenda-se análise adicional',
    colorKey: 'yellow',
    rgb: [234, 179, 8],
  },
  suspect: {
    label: 'SUSPEITO',
    description: 'Consistente com manipulação',
    colorKey: 'orange',
    rgb: [249, 115, 22],
  },
  strong: {
    label: 'MANIPULAÇÃO PROVÁVEL',
    description: 'Manipulação altamente provável',
    colorKey: 'red',
    rgb: [239, 68, 68],
  },
};

export function classifyScore(score: number | null | undefined): Verdict {
  const s = typeof score === 'number' && !Number.isNaN(score) ? score : 0;
  let level: VerdictLevel;
  if (s < T_INCONCLUSIVE) level = 'clean';
  else if (s < T_SUSPECT) level = 'inconclusive';
  else if (s < T_STRONG)  level = 'suspect';
  else                    level = 'strong';
  return {
    level,
    isFake: level === 'suspect' || level === 'strong',
    ...VERDICTS[level],
  };
}

/** Tailwind text colour class for a verdict (e.g. 'text-red-400'). */
export function verdictTextClass(v: Verdict): string {
  return {
    green:  'text-green-400',
    yellow: 'text-yellow-400',
    orange: 'text-orange-400',
    red:    'text-red-400',
  }[v.colorKey];
}

/** Tailwind background class (subdued) for a verdict card. */
export function verdictBgClass(v: Verdict): string {
  return {
    green:  'bg-green-500/10 border-l-green-500',
    yellow: 'bg-yellow-500/10 border-l-yellow-500',
    orange: 'bg-orange-500/10 border-l-orange-500',
    red:    'bg-red-500/10 border-l-red-500',
  }[v.colorKey];
}
