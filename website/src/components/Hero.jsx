import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import ExampleResultCard from './landing/ExampleResultCard';
import { fadeUp, staggerContainer } from './analyzer/motion';
import { BUILT_ON } from './landing/data';

const FACTS = [
  ['6', 'parallel checks'],
  ['4', 'news APIs'],
  ['~5s', 'typical analysis'],
];

const Hero = () => (
  <section aria-labelledby="hero-title" className="relative isolate">
    {/* Faint grid + radial wash (decorative) */}
    <div className="landing-grid-wash pointer-events-none absolute inset-0 -z-10" aria-hidden="true" />

    <div className="mx-auto grid max-w-6xl items-center gap-12 px-4 pb-16 pt-10 sm:px-6 sm:pt-16 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)] lg:gap-16 lg:pb-24 lg:pt-20">
      <motion.div variants={staggerContainer} initial="hidden" animate="show" className="min-w-0">
        <motion.p variants={fadeUp}>
          <span className="inline-flex h-8 items-center gap-2 rounded-full border border-zinc-200/70 bg-white/80 px-3 text-xs font-medium text-zinc-600 shadow-sm">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" aria-hidden="true" />
            Transformer + classical ML + live sources
          </span>
        </motion.p>

        <motion.h1
          variants={fadeUp}
          id="hero-title"
          className="mt-6 text-[44px] font-semibold leading-[1.02] tracking-[-0.04em] text-zinc-950 sm:text-[56px] lg:text-[72px]"
        >
          Check the claim
          <span className="block text-zinc-500">before you share it.</span>
        </motion.h1>

        <motion.p variants={fadeUp} className="mt-6 max-w-xl text-base leading-7 text-zinc-600 sm:text-lg sm:leading-8">
          News-Check runs six checks in parallel, from a fine-tuned DistilBERT model to Wikipedia and four
          news APIs, then weighs them by input type into a clear verdict with the evidence behind it.
        </motion.p>

        <motion.div variants={fadeUp} className="mt-8 flex flex-col gap-3 sm:flex-row">
          <Link
            to="/analyze"
            className="inline-flex min-h-12 items-center justify-center gap-2 rounded-xl bg-zinc-900 px-5 text-[15px] font-medium text-white shadow-card transition-colors hover:bg-zinc-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900/40 focus-visible:ring-offset-2"
          >
            Try the analyzer
            <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </Link>
          <a
            href="#how-it-works"
            className="inline-flex min-h-12 items-center justify-center rounded-xl border border-zinc-200/70 bg-white px-5 text-[15px] font-medium text-zinc-800 shadow-sm transition-colors hover:border-zinc-300 hover:bg-zinc-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900/40 focus-visible:ring-offset-2"
          >
            How it works
          </a>
        </motion.div>

        <motion.dl variants={fadeUp} className="mt-10 grid max-w-md grid-cols-3 divide-x divide-zinc-200/70 border-y border-zinc-200/70">
          {FACTS.map(([value, label]) => (
            <div key={label} className="flex flex-col-reverse px-3 py-3 first:pl-0">
              <dt className="mt-1 text-xs text-zinc-500">{label}</dt>
              <dd className="text-xl font-semibold tracking-[-0.02em] text-zinc-950 tabular-nums">{value}</dd>
            </div>
          ))}
        </motion.dl>
      </motion.div>

      <motion.div variants={fadeUp} initial="hidden" animate="show" className="min-w-0">
        <ExampleResultCard />
      </motion.div>
    </div>

    {/* Built-on row */}
    <div className="mx-auto max-w-6xl px-4 pb-16 sm:px-6 lg:pb-20">
      <div className="flex flex-col items-start gap-3 border-t border-zinc-200/70 pt-6 sm:flex-row sm:items-center sm:gap-5">
        <p className="micro-label flex-shrink-0">Built on</p>
        <ul className="flex flex-wrap gap-2" aria-label="Models and data sources">
          {BUILT_ON.map((name) => (
            <li
              key={name}
              className="inline-flex h-8 items-center rounded-full border border-zinc-200/70 bg-white px-3 text-[13px] font-medium text-zinc-600"
            >
              {name}
            </li>
          ))}
        </ul>
      </div>
    </div>
  </section>
);

export default Hero;
