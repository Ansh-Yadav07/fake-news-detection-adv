import React from 'react';
import { motion } from 'framer-motion';
import Meter from './Meter';
import { EASE_OUT } from './motion';
import { isUnknownModel } from './verdictTheme';

// `verificationWeights` from decisionLogic are per-signal REAL-support scores
// (0-100). External signals report 0 when they were unavailable.
const SIGNALS = [
  { key: 'wikipedia', label: 'Wikipedia', color: '#8b5cf6', optional: true }, // violet-500
  { key: 'gnews', label: 'News', color: '#0ea5e9', optional: true }, // sky-500
  { key: 'ml', label: 'ML models', color: '#3f3f46', optional: true }, // zinc-700
  { key: 'credibility', label: 'Credibility', color: '#10b981', optional: true }, // emerald-500
  { key: 'linguistic', label: 'Linguistic', color: '#f59e0b' }, // amber-500
  { key: 'clickbait', label: 'Clickbait-free', color: '#fb7185' }, // rose-400
];

const qualify = (pct) => (pct >= 75 ? 'High' : pct >= 45 ? 'Moderate' : 'Low');

const MeterRow = ({ label, valueText, percent, qualifier, color, unavailable, note }) => (
  <div>
    <div className="mb-2 flex items-baseline justify-between gap-3">
      <span className="text-[13px] font-medium text-zinc-700">{label}</span>
      {unavailable ? (
        <span className="text-[13px] font-medium text-zinc-400">Unavailable</span>
      ) : (
        <span className="text-[13px] tabular-nums text-zinc-500">
          <span className="font-semibold text-zinc-900">{valueText}</span>
          <span className="ml-1.5">{qualifier}</span>
        </span>
      )}
    </div>
    <Meter percent={percent} color={color} label={label} valueText={valueText} unavailable={unavailable} />
    {note && <p className="mt-2 text-xs text-zinc-500">{note}</p>}
  </div>
);

const EvidenceBreakdown = ({ verificationWeights, agreementScore, robustnessScore, transformerLabel, hybridLabel, accent }) => {
  const signals = verificationWeights
    ? SIGNALS.map((s) => ({ ...s, value: Math.max(0, Number(verificationWeights[s.key]) || 0) }))
    : [];
  // Only positive signals get a segment; widths are normalised to their sum so
  // the bar stays proportional to the legend values.
  const segments = signals.filter((s) => s.value > 0);
  const total = segments.reduce((sum, s) => sum + s.value, 0);

  const agreement = Number(agreementScore);
  const robustness = Number(robustnessScore);
  const modelsMissing = isUnknownModel(transformerLabel) || isUnknownModel(hybridLabel);
  const agreementUnavailable = !Number.isFinite(agreement) || modelsMissing;
  const robustnessUnavailable = !Number.isFinite(robustness);
  const agreementPct = Math.min(100, Math.max(0, agreement || 0));
  const robustnessPct = Math.min(100, Math.max(0, (robustness || 0) * 10));

  return (
    <div className="grid gap-8 lg:grid-cols-12 lg:gap-10">
      {/* Signal mix */}
      <div className="min-w-0 lg:col-span-7">
        <div className="mb-4">
          <h4 className="micro-label">Evidence breakdown</h4>
          <p className="mt-1 text-xs leading-5 text-zinc-500">
            How strongly each signal supports authenticity (0–100). Bar widths are relative.
          </p>
        </div>

        {segments.length > 0 ? (
          <>
            <motion.div
              className="flex h-2.5 origin-left gap-[3px] overflow-hidden rounded-full"
              role="img"
              aria-label={`Evidence by signal: ${segments.map((s) => `${s.label} ${s.value}`).join(', ')}`}
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              transition={{ duration: 0.9, ease: EASE_OUT, delay: 0.2 }}
            >
              {segments.map((s) => (
                <div
                  key={s.key}
                  className="h-full"
                  style={{ width: `${(s.value / total) * 100}%`, backgroundColor: s.color }}
                  title={`${s.label}: ${s.value}`}
                />
              ))}
            </motion.div>

            <ul className="mt-6 grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-3" aria-label="Signal values">
              {signals.map((s) => {
                const inactive = s.value === 0;
                return (
                  <li key={s.key} className="flex min-w-0 items-center gap-2">
                    <span
                      className="h-2.5 w-2.5 flex-shrink-0 rounded-[3px]"
                      style={{ backgroundColor: inactive ? '#e4e4e7' : s.color }}
                      aria-hidden="true"
                    />
                    <span className={`flex-1 truncate text-[13px] ${inactive ? 'text-zinc-400' : 'text-zinc-600'}`}>
                      {s.label}
                    </span>
                    <span className={`text-[13px] font-semibold tabular-nums ${inactive ? 'text-zinc-400' : 'text-zinc-900'}`}>
                      {inactive && s.optional ? <><span aria-hidden="true">—</span><span className="sr-only">not available</span></> : s.value}
                    </span>
                  </li>
                );
              })}
            </ul>
          </>
        ) : (
          <div className="flex h-10 items-center rounded-xl border border-dashed border-zinc-200 px-4 text-[13px] text-zinc-500">
            Signal breakdown unavailable for this result.
          </div>
        )}
      </div>

      {/* Agreement + robustness */}
      <div className="grid content-start gap-6 sm:grid-cols-2 lg:col-span-5 lg:grid-cols-1">
        <MeterRow
          label="Model agreement"
          percent={agreementPct}
          valueText={`${Math.round(agreementPct)}%`}
          qualifier={qualify(agreementPct)}
          color={accent}
          unavailable={agreementUnavailable}
          note={agreementUnavailable && modelsMissing ? 'Requires both ML models; at least one was unavailable.' : null}
        />
        <MeterRow
          label="Robustness"
          percent={robustnessPct}
          valueText={`${(Math.round((robustness || 0) * 10) / 10).toString()}/10`}
          qualifier={qualify(robustnessPct)}
          color={accent}
          unavailable={robustnessUnavailable}
        />
      </div>
    </div>
  );
};

export default EvidenceBreakdown;
