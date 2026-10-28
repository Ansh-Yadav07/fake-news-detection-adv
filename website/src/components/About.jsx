import React from 'react';
import { Reveal, RevealItem, SectionHeader } from './landing/Reveal';
import { STEPS } from './landing/data';

/** "How it works": the four-stage analysis flow. */
const About = () => (
  <section id="how-it-works" aria-labelledby="how-it-works-title" className="scroll-mt-20 border-t border-zinc-200/70 bg-white">
    <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6 lg:py-28">
      <SectionHeader id="how-it-works-title" eyebrow="How it works" title="From text to verdict in four steps">
        Every submission follows the same pipeline: classify the input, run every check at once, cross-verify
        against live sources, then weigh the evidence.
      </SectionHeader>

      <div className="relative mt-14 lg:mt-20">
        {/* Desktop connector through the step markers (decorative) */}
        <div
          className="pointer-events-none absolute left-5 right-[calc(25%-2.375rem)] top-5 hidden h-px bg-zinc-200 lg:block"
          aria-hidden="true"
        />
        <Reveal as="ol" stagger className="relative grid lg:grid-cols-4 lg:gap-6">
        {STEPS.map((step, i) => {
          const { Icon } = step;
          const last = i === STEPS.length - 1;
          return (
            <RevealItem as="li" key={step.title} className="relative flex gap-5 pb-10 last:pb-0 lg:block lg:pb-0">
              {/* Mobile vertical connector (decorative) */}
              {!last && (
                <span className="absolute bottom-0 left-5 top-12 w-px bg-zinc-200 lg:hidden" aria-hidden="true" />
              )}
              <span
                className="relative z-10 flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full border border-zinc-200/70 bg-white text-sm font-semibold text-zinc-900 shadow-card tabular-nums"
                aria-hidden="true"
              >
                {String(i + 1).padStart(2, '0')}
              </span>
              <div className="min-w-0 pt-1.5 lg:mt-6 lg:pt-0">
                <div className="flex items-center gap-2">
                  <Icon className="h-4 w-4 text-zinc-400" aria-hidden="true" />
                  <h3 className="text-[17px] font-semibold tracking-[-0.01em] text-zinc-950">
                    <span className="sr-only">Step {i + 1}: </span>
                    {step.title}
                  </h3>
                </div>
                <p className="mt-2 text-sm leading-6 text-zinc-600">{step.desc}</p>
              </div>
            </RevealItem>
          );
        })}
        </Reveal>
      </div>
    </div>
  </section>
);

export default About;
