import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { Reveal } from './Reveal';

const CtaBand = () => (
  <section aria-labelledby="cta-title" className="border-t border-zinc-200/70">
    <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6 lg:py-24">
      <Reveal className="relative isolate overflow-hidden rounded-3xl border border-zinc-200/70 bg-white px-6 py-12 text-center shadow-hero sm:px-12 sm:py-16">
        <div className="landing-grid-wash pointer-events-none absolute inset-0 -z-10 opacity-70" aria-hidden="true" />
        <h2 id="cta-title" className="mx-auto max-w-xl text-[30px] font-semibold leading-[1.1] tracking-[-0.03em] text-zinc-950 sm:text-[40px]">
          Have a headline you are not sure about?
        </h2>
        <p className="mx-auto mt-4 max-w-md text-base leading-7 text-zinc-600">
          Paste a claim or an article and get a verdict with its evidence in seconds.
        </p>
        <Link
          to="/analyze"
          className="mt-8 inline-flex min-h-12 items-center justify-center gap-2 rounded-xl bg-zinc-900 px-5 text-[15px] font-medium text-white shadow-card transition-colors hover:bg-zinc-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900/40 focus-visible:ring-offset-2"
        >
          Open the analyzer
          <ArrowRight className="h-4 w-4" aria-hidden="true" />
        </Link>
      </Reveal>
    </div>
  </section>
);

export default CtaBand;
