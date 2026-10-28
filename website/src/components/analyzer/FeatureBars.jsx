import React from 'react';
import AnalysisCard from './AnalysisCard';
import Meter from './Meter';
import { AlignLeft } from 'lucide-react';

// Severity tint for "warning" signals: higher = more sensational
const severity = (pct) => {
  if (pct > 70) return { label: 'High', color: '#ef4444', text: 'text-red-600' };
  if (pct > 40) return { label: 'Moderate', color: '#f59e0b', text: 'text-amber-600' };
  return { label: 'Low', color: '#a1a1aa', text: 'text-zinc-500' };
};

const SignalRow = ({ label, value, max = 100, isWarning = true }) => {
  const safe = Number(value);
  const finite = Number.isFinite(safe);
  const pct = finite ? Math.min(100, Math.max(0, (safe / max) * 100)) : 0;
  const sev = isWarning ? severity(pct) : { label: null, color: '#3f3f46', text: 'text-zinc-500' };
  const display = max === 100 ? `${Math.round(safe)}%` : `${Math.round(safe * 10) / 10}/${max}`;

  return (
    <li>
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <span className="text-[13px] font-medium text-zinc-700">{label}</span>
        {finite ? (
          <span className="text-[13px] tabular-nums">
            <span className="font-semibold text-zinc-900">{display}</span>
            {sev.label && <span className={`ml-1.5 ${sev.text}`}>{sev.label}</span>}
          </span>
        ) : (
          <span className="text-[13px] font-medium text-zinc-400">Unavailable</span>
        )}
      </div>
      <Meter percent={pct} color={sev.color} label={label} valueText={finite ? display : undefined} unavailable={!finite} />
    </li>
  );
};

const FeatureBars = ({ punctuation, uppercase, complexity, clickbait }) => (
  <AnalysisCard title="Linguistic signals" subtitle="Style markers common in misleading content" icon={AlignLeft}>
    <ul className="space-y-6">
      <SignalRow label="Clickbait" value={clickbait} />
      <SignalRow label="Punctuation density" value={punctuation} />
      <SignalRow label="Uppercase ratio" value={uppercase} />
      <SignalRow label="Complexity" value={complexity} max={10} isWarning={false} />
    </ul>
  </AnalysisCard>
);

export default FeatureBars;
