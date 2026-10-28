import React from 'react';
import { Scale } from 'lucide-react';
import { Reveal, RevealItem, SectionHeader } from './landing/Reveal';
import { CHECKS, SIGNAL_COLORS, SIGNAL_LABELS, WEIGHT_PROFILES } from './landing/data';

const Tag = ({ children }) => (
  <li className="inline-flex h-6 items-center rounded-md border border-zinc-200/70 bg-zinc-50 px-2 text-[11px] font-medium text-zinc-600">
    {children}
  </li>
);

const CheckCard = ({ check }) => {
  const { Icon } = check;
  return (
    <RevealItem
      as="li"
      className="flex flex-col rounded-3xl border border-zinc-200/70 bg-white p-6 shadow-card md:last:col-span-2 lg:last:col-span-1"
    >
      <span className={`flex h-10 w-10 items-center justify-center rounded-xl ring-1 ${check.tile}`} aria-hidden="true">
        <Icon className="h-5 w-5" />
      </span>
      <h3 className="mt-5 text-[17px] font-semibold tracking-[-0.01em] text-zinc-950">{check.title}</h3>
      <p className="mt-2 flex-1 text-sm leading-6 text-zinc-600">{check.desc}</p>
      <ul className="mt-5 flex flex-wrap gap-1.5" aria-label="Tags">
        {check.tags.map((t) => (
          <Tag key={t}>{t}</Tag>
        ))}
      </ul>
    </RevealItem>
  );
};

const WeightBar = ({ profile }) => {
  const { Icon } = profile;
  const summary = profile.weights.map(([k, w]) => `${SIGNAL_LABELS[k]} ${w}%`).join(', ');
  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <Icon className="h-4 w-4 text-zinc-400" aria-hidden="true" />
        <h4 className="text-[13px] font-semibold text-zinc-800">{profile.label}</h4>
      </div>
      <div className="flex h-2.5 gap-[3px] overflow-hidden rounded-full" role="img" aria-label={`${profile.label} weights: ${summary}`}>
        {profile.weights.map(([k, w]) => (
          <span key={k} className="h-full" style={{ width: `${w}%`, backgroundColor: SIGNAL_COLORS[k] }} />
        ))}
      </div>
      <ul className="mt-2.5 flex flex-wrap gap-x-4 gap-y-1.5" aria-hidden="true">
        {profile.weights.map(([k, w]) => (
          <li key={k} className="inline-flex items-center gap-1.5 text-xs text-zinc-600">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: SIGNAL_COLORS[k] }} />
            {SIGNAL_LABELS[k]}
            <span className="font-semibold tabular-nums text-zinc-900">{w}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
};

const VerdictEngineCard = () => (
  <RevealItem
    as="li"
    className="relative isolate overflow-hidden rounded-3xl border border-zinc-200/70 bg-white p-6 shadow-hero sm:p-8 md:col-span-2 lg:row-span-2"
  >
    <div
      className="pointer-events-none absolute inset-0 -z-10"
      aria-hidden="true"
      style={{ backgroundImage: 'radial-gradient(70% 60% at 100% 0%, rgb(16 185 129 / 0.07) 0%, transparent 65%)' }}
    />
    <div className="flex items-start justify-between gap-4">
      <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-50 text-emerald-700 ring-1 ring-emerald-600/15" aria-hidden="true">
        <Scale className="h-5 w-5" />
      </span>
      <span className="micro-label">Check 6 · Decision</span>
    </div>
    <h3 className="mt-5 text-[22px] font-semibold tracking-[-0.02em] text-zinc-950 sm:text-2xl">Input-aware verdict engine</h3>
    <p className="mt-2 max-w-lg text-sm leading-6 text-zinc-600">
      Weights shift with the input type, so fact claims lean on Wikipedia and news leans on live coverage.
      Missing signals are renormalised away, and override rules handle contradictions and trusted-source matches.
    </p>

    <div className="mt-7 space-y-6 border-t border-zinc-200/70 pt-6">
      <p className="micro-label">Base weight profiles</p>
      {WEIGHT_PROFILES.map((p) => (
        <WeightBar key={p.id} profile={p} />
      ))}
    </div>
  </RevealItem>
);

const Features = () => (
  <section id="features" aria-labelledby="features-title" className="scroll-mt-20 border-t border-zinc-200/70">
    <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6 lg:py-28">
      <SectionHeader id="features-title" eyebrow="Features" title="Six checks, one transparent verdict">
        Deep learning, classical ML, style analysis and live cross-referencing, each reported separately so
        you can see why a verdict was reached.
      </SectionHeader>

      <Reveal as="ul" stagger className="mt-14 grid gap-4 md:grid-cols-2 lg:mt-16 lg:grid-cols-3 lg:gap-5">
        <VerdictEngineCard />
        {CHECKS.map((check) => (
          <CheckCard key={check.title} check={check} />
        ))}
      </Reveal>
    </div>
  </section>
);

export default Features;
