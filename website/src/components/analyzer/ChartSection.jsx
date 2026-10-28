import React from 'react';
import AnalysisCard from './AnalysisCard';
import Meter from './Meter';
import { Cpu } from 'lucide-react';
import { isUnknownModel } from './verdictTheme';

const PREDICTION_STYLES = {
  REAL: { pill: 'bg-emerald-50 text-emerald-700 ring-emerald-600/15', color: '#059669' },
  FAKE: { pill: 'bg-red-50 text-red-700 ring-red-600/15', color: '#dc2626' },
};
const NEUTRAL = { pill: 'bg-zinc-100 text-zinc-600 ring-zinc-500/15', color: '#52525b' };

const ModelRow = ({ name, description, label, confidence }) => {
  const unavailable = isUnknownModel(label);
  const prediction = unavailable ? null : String(label).trim().toUpperCase();
  const style = PREDICTION_STYLES[prediction] || NEUTRAL;
  const pct = Math.round(Math.min(100, Math.max(0, Number(confidence) || 0)));

  return (
    <li className="py-4 first:pt-0 last:pb-0">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-sm font-medium text-zinc-900">{name}</p>
          <p className="mt-0.5 text-xs text-zinc-500">{description}</p>
        </div>
        {unavailable ? (
          <span className="text-sm font-medium text-zinc-400">Unavailable</span>
        ) : (
          <span className="flex items-start font-semibold leading-none tracking-tight text-zinc-950 tabular-nums">
            <span className="text-2xl">{pct}</span>
            <span className="ml-0.5 mt-0.5 text-sm text-zinc-400">%</span>
          </span>
        )}
      </div>
      <div className="mt-3 flex items-center gap-3">
        <Meter
          className="flex-1"
          percent={pct}
          color={style.color}
          label={`${name} confidence`}
          valueText={unavailable ? undefined : `${pct}% ${prediction}`}
          unavailable={unavailable}
        />
        <span
          className={`inline-flex h-6 min-w-[3.5rem] items-center justify-center rounded-md px-2 text-[11px] font-semibold uppercase tracking-[0.06em] ring-1 ${style.pill}`}
        >
          {unavailable ? 'Offline' : prediction}
        </span>
      </div>
    </li>
  );
};

const ChartSection = ({ transformerConf, hybridConf, transformerLabel, hybridLabel }) => (
  <AnalysisCard title="Model confidence" subtitle="Each model's certainty in its own prediction" icon={Cpu}>
    <ul className="divide-y divide-zinc-200/70">
      <ModelRow name="Transformer" description="Fine-tuned language model" label={transformerLabel} confidence={transformerConf} />
      <ModelRow name="Hybrid ML" description="Logistic regression on text features" label={hybridLabel} confidence={hybridConf} />
    </ul>
  </AnalysisCard>
);

export default ChartSection;
