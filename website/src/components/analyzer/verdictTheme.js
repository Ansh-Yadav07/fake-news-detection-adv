import {
  AlertCircle,
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  HelpCircle,
  ShieldAlert,
  ShieldCheck,
} from 'lucide-react';

/**
 * Visual tone for every verdict label decisionLogic.js can return.
 *
 * - `rgb` is a space-separated triplet used for the aurora wash, glow and
 *   gauge stroke (`rgb(${rgb} / <alpha>)`).
 * - Class strings are written out literally so Tailwind's JIT picks them up.
 */
export const VERDICT_TONES = {
  'VERIFIED FACT': {
    display: 'Verified fact',
    tagline: 'Confirmed against established reference sources.',
    Icon: BookOpen,
    rgb: '4 120 87', // emerald-700
    text: 'text-emerald-700',
    tile: 'bg-emerald-50 text-emerald-700 ring-emerald-600/15',
  },
  VERIFIED: {
    display: 'Verified',
    tagline: 'Corroborated by independent online evidence.',
    Icon: ShieldCheck,
    rgb: '5 150 105', // emerald-600
    text: 'text-emerald-700',
    tile: 'bg-emerald-50 text-emerald-700 ring-emerald-600/15',
  },
  'LIKELY REAL': {
    display: 'Likely real',
    tagline: 'Signals lean authentic, with partial corroboration.',
    Icon: CheckCircle2,
    rgb: '2 132 199', // sky-600
    text: 'text-sky-700',
    tile: 'bg-sky-50 text-sky-700 ring-sky-600/15',
  },
  REAL: {
    display: 'Real',
    tagline: 'Model and style signals indicate authentic content.',
    Icon: CheckCircle2,
    rgb: '5 150 105', // emerald-600
    text: 'text-emerald-700',
    tile: 'bg-emerald-50 text-emerald-700 ring-emerald-600/15',
  },
  UNVERIFIED: {
    display: 'Unverified',
    tagline: 'Not enough independent evidence to confirm or refute.',
    Icon: HelpCircle,
    rgb: '217 119 6', // amber-600
    text: 'text-amber-700',
    tile: 'bg-amber-50 text-amber-700 ring-amber-600/15',
  },
  UNCERTAIN: {
    display: 'Uncertain',
    tagline: 'Signals are mixed; treat this content with care.',
    Icon: HelpCircle,
    rgb: '113 113 122', // zinc-500
    text: 'text-zinc-600',
    tile: 'bg-zinc-100 text-zinc-600 ring-zinc-500/15',
  },
  SUSPICIOUS: {
    display: 'Suspicious',
    tagline: 'Several warning signs; verify before sharing.',
    Icon: AlertTriangle,
    rgb: '234 88 12', // orange-600
    text: 'text-orange-700',
    tile: 'bg-orange-50 text-orange-700 ring-orange-600/15',
  },
  'LIKELY FAKE': {
    display: 'Likely fake',
    tagline: 'Evidence and signals point toward misinformation.',
    Icon: ShieldAlert,
    rgb: '220 38 38', // red-600
    text: 'text-red-700',
    tile: 'bg-red-50 text-red-700 ring-red-600/15',
  },
  FAKE: {
    display: 'Fake',
    tagline: 'Strong indicators of fabricated or misleading content.',
    Icon: AlertCircle,
    rgb: '185 28 28', // red-700
    text: 'text-red-700',
    tile: 'bg-red-50 text-red-700 ring-red-600/15',
  },
};

const FALLBACK_TONE = {
  tagline: 'This verdict could not be categorised.',
  Icon: HelpCircle,
  rgb: '113 113 122',
  text: 'text-zinc-600',
  tile: 'bg-zinc-100 text-zinc-600 ring-zinc-500/15',
};

const toTitle = (value) =>
  String(value)
    .toLowerCase()
    .replace(/[_-]+/g, ' ')
    .replace(/^\w/, (c) => c.toUpperCase());

/** Resolve a verdict label to its tone; unknown labels render neutrally. */
export const resolveVerdict = (label) => {
  const key = typeof label === 'string' ? label.trim().toUpperCase() : '';
  if (VERDICT_TONES[key]) return { key, ...VERDICT_TONES[key] };
  return { key, ...FALLBACK_TONE, display: key ? toTitle(key) : 'Unknown' };
};

export const rgba = (rgb, alpha) => `rgb(${rgb} / ${alpha})`;

/** Is a model label missing/unknown (degraded backend)? */
export const isUnknownModel = (label) =>
  !label || String(label).trim().toUpperCase() === 'UNKNOWN';

export const clampPct = (value) => Math.min(100, Math.max(0, Number(value) || 0));
