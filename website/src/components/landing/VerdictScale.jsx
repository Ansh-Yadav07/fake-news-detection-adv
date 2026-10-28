import React from 'react';
import { Reveal, RevealItem, SectionHeader } from './Reveal';
import { VERDICT_GROUPS } from './data';
import { resolveVerdict, rgba } from '../analyzer/verdictTheme';

const VerdictScale = () => (
  <section id="verdicts" aria-labelledby="verdicts-title" className="scroll-mt-20 border-t border-zinc-200/70 bg-white">
    <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6 lg:py-28">
      <SectionHeader id="verdicts-title" eyebrow="Verdicts" title="More nuance than real or fake">
        Every result lands on one of these labels. Too little evidence reads as unverified, never as
        automatically fake.
      </SectionHeader>

      <Reveal stagger className="mt-14 grid gap-4 lg:mt-16 lg:grid-cols-3 lg:gap-5">
        {VERDICT_GROUPS.map((group) => {
          const tones = group.keys.map(resolveVerdict);
          return (
            <RevealItem key={group.id} className="overflow-hidden rounded-3xl border border-zinc-200/70 bg-white shadow-card">
              {/* Tone strip (decorative) */}
              <div className="flex h-1.5" aria-hidden="true">
                {tones.map((t) => (
                  <span key={t.key} className="flex-1" style={{ backgroundColor: rgba(t.rgb, 0.85) }} />
                ))}
              </div>
              <div className="p-6">
                <h3 className="micro-label">{group.title}</h3>
                <ul className="mt-4 space-y-4">
                  {tones.map((t) => {
                    const { Icon } = t;
                    return (
                      <li key={t.key} className="flex gap-3">
                        <span
                          className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg ring-1 ${t.tile}`}
                          aria-hidden="true"
                        >
                          <Icon className="h-4 w-4" />
                        </span>
                        <div className="min-w-0">
                          <p className={`text-[15px] font-semibold ${t.text}`}>{t.display}</p>
                          <p className="mt-0.5 text-[13px] leading-5 text-zinc-600">{t.tagline}</p>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              </div>
            </RevealItem>
          );
        })}
      </Reveal>
    </div>
  </section>
);

export default VerdictScale;
