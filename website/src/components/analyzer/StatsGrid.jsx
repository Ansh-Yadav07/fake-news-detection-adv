import React from 'react';
import AnalysisCard from './AnalysisCard';
import { Type } from 'lucide-react';

const formatNumber = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? n.toLocaleString() : '—';
};

const Stat = ({ label, value, unit }) => (
  <div className="bg-white p-4">
    <dt className="micro-label">{label}</dt>
    <dd className="mt-2 flex items-baseline gap-1 text-2xl font-semibold leading-none tracking-tight text-zinc-950 tabular-nums">
      {value}
      {unit && <span className="text-sm font-medium tracking-normal text-zinc-400">{unit}</span>}
    </dd>
  </div>
);

const StatsGrid = ({ wordCount, avgWordLength, upperRatio, punctDensity }) => (
  <AnalysisCard title="Text statistics" subtitle="Measured from the submitted text" icon={Type}>
    {/* gap-px over a tinted background draws 1px hairlines between tiles */}
    <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-zinc-200/70 bg-zinc-200/70 sm:grid-cols-4">
      <Stat label="Words" value={formatNumber(wordCount)} />
      <Stat label="Avg word" value={formatNumber(avgWordLength)} unit="chars" />
      <Stat label="Uppercase" value={formatNumber(upperRatio)} unit="%" />
      <Stat label="Punctuation" value={formatNumber(punctDensity)} unit="%" />
    </dl>
  </AnalysisCard>
);

export default StatsGrid;
