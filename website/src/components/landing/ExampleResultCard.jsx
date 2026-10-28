import React from 'react';
import { BookOpen, Timer } from 'lucide-react';
import Ring from '../analyzer/Ring';
import Meter from '../analyzer/Meter';
import { resolveVerdict, rgba } from '../analyzer/verdictTheme';
import { SIGNAL_COLORS } from './data';

// Illustrative sample only: a relational claim that Wikipedia contradiction
// detection catches. 85% matches the contradiction override in the README.
const SAMPLE = {
  text: 'Sydney is the capital of Australia.',
  verdict: 'LIKELY FAKE',
  confidence: 85,
  reason: 'Wikipedia contradiction: Canberra is the capital of Australia.',
  signals: [
    { label: 'Wikipedia', value: 6, color: SIGNAL_COLORS.wikipedia },
    { label: 'ML models', value: 38, color: SIGNAL_COLORS.ml },
    { label: 'Linguistic', value: 72, color: SIGNAL_COLORS.linguistic },
  ],
};

const ExampleResultCard = () => {
  const tone = resolveVerdict(SAMPLE.verdict);
  const { Icon } = tone;

  return (
    <figure className="relative mx-auto w-full max-w-md lg:max-w-none">
      {/* Offset backing sheet (decorative) */}
      <div
        className="absolute inset-x-6 -bottom-3 top-6 rounded-3xl border border-zinc-200/70 bg-white/60 shadow-card"
        aria-hidden="true"
      />

      <div className="relative isolate overflow-hidden rounded-3xl border border-zinc-200/70 bg-white shadow-hero">
        <div
          className="pointer-events-none absolute inset-0 -z-10"
          aria-hidden="true"
          style={{
            backgroundImage: `radial-gradient(80% 70% at 0% 0%, ${rgba(tone.rgb, 0.09)} 0%, transparent 60%)`,
          }}
        />

        <div className="p-5 sm:p-6">
          <div className="flex items-center justify-between gap-3">
            <span className="inline-flex h-7 items-center rounded-full bg-zinc-900 px-2.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-white">
              Example result
            </span>
            <span className="inline-flex h-7 items-center gap-1.5 rounded-full border border-zinc-200/70 bg-white/80 px-2.5 text-xs font-medium text-zinc-600">
              <BookOpen className="h-3.5 w-3.5 text-zinc-400" aria-hidden="true" />
              Fact claim
            </span>
          </div>

          <blockquote className="mt-5 border-l-2 pl-3 text-sm leading-6 text-zinc-600" style={{ borderColor: rgba(tone.rgb, 0.35) }}>
            “{SAMPLE.text}”
          </blockquote>

          <div className="mt-5 flex items-center justify-between gap-4">
            <div className="min-w-0">
              <p className={`micro-label ${tone.text}`}>Final verdict</p>
              <p className="mt-2 flex items-center gap-2.5 text-[26px] font-semibold leading-tight tracking-[-0.03em] text-zinc-950">
                <span
                  className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl ring-1 ${tone.tile}`}
                  aria-hidden="true"
                >
                  <Icon className="h-5 w-5" />
                </span>
                {tone.display}
              </p>
            </div>
            <Ring
              value={SAMPLE.confidence}
              label={`Example confidence ${SAMPLE.confidence}%`}
              color={rgba(tone.rgb, 1)}
              trackColor={rgba(tone.rgb, 0.1)}
              strokeWidth={8}
              sizeClass="h-20 w-20"
            >
              <span className="flex items-start text-[22px] font-semibold leading-none tracking-[-0.04em] text-zinc-950 tabular-nums">
                {SAMPLE.confidence}
                <span className="mt-0.5 text-xs text-zinc-400">%</span>
              </span>
            </Ring>
          </div>

          <p className="mt-4 text-sm leading-6 text-zinc-700">{SAMPLE.reason}</p>
        </div>

        <div className="border-t border-zinc-200/70 bg-zinc-50/50 p-5 sm:p-6">
          <p className="micro-label">Evidence breakdown</p>
          <div className="mt-3 space-y-3">
            {SAMPLE.signals.map((s) => (
              <div key={s.label}>
                <div className="mb-1.5 flex items-baseline justify-between text-[13px]">
                  <span className="font-medium text-zinc-700">{s.label}</span>
                  <span className="font-semibold tabular-nums text-zinc-900">{s.value}</span>
                </div>
                <Meter percent={s.value} color={s.color} label={`Example ${s.label} support`} valueText={`${s.value} of 100`} size="sm" />
              </div>
            ))}
          </div>
          <p className="mt-4 flex items-center gap-1.5 text-xs text-zinc-500">
            <Timer className="h-3.5 w-3.5 text-zinc-400" aria-hidden="true" />
            Illustrative sample data · 6 parallel checks
          </p>
        </div>
      </div>
      <figcaption className="sr-only">Illustrative example of an analysis result using sample data.</figcaption>
    </figure>
  );
};

export default ExampleResultCard;
