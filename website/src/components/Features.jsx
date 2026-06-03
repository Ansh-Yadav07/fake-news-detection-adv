import React, { useEffect, useRef, useState } from 'react';
import {
  Newspaper, ShieldAlert, Zap, Globe, Award,
  ScanSearch, BarChart3, AlertTriangle, Sparkles
} from 'lucide-react';

const useInView = (options = {}) => {
  const ref = useRef(null);
  const [isInView, setIsInView] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setIsInView(true); },
      { threshold: 0.15, ...options }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  return [ref, isInView];
};

const featuresList = [
  {
    title: 'Real-Time News Cross-Referencing',
    desc: 'Searches GNews, NewsData.io, NewsAPI.org in parallel — with SerpAPI (Google News) as a smart fallback when primary sources return insufficient results.',
    icon: Newspaper,
    accent: 'bg-emerald-50 text-emerald-600 border-emerald-200',
  },
  {
    title: 'Contradiction Detection',
    desc: 'Wikipedia verification doesn\'t just match — it actively detects when your claim contradicts established facts, like wrong capitals or debunked theories.',
    icon: AlertTriangle,
    accent: 'bg-rose-50 text-rose-600 border-rose-200',
  },
  {
    title: 'Input-Aware Verification',
    desc: 'Automatically classifies input as a fact claim, news headline, or mixed content — then dynamically adjusts which verification sources carry the most weight.',
    icon: ScanSearch,
    accent: 'bg-violet-50 text-violet-600 border-violet-200',
  },
  {
    title: 'Trusted Source Scoring',
    desc: 'Articles from Reuters, BBC, AP News, The Hindu, NDTV, and other trusted outlets receive priority weighting — a single BBC confirmation outweighs multiple unknown sources.',
    icon: Award,
    accent: 'bg-amber-50 text-amber-600 border-amber-200',
  },
  {
    title: 'Parallel Processing Engine',
    desc: 'All 6 analysis tasks — transformer, ML model, linguistic analysis, Wikipedia, news APIs, and verdict computation — execute concurrently in ~5 seconds.',
    icon: Zap,
    accent: 'bg-sky-50 text-sky-600 border-sky-200',
  },
  {
    title: 'Clickbait & Sensationalism Scoring',
    desc: 'Continuous 0–100% clickbait probability based on trigger words, excessive punctuation, ALL CAPS patterns, and writing style heuristics — not a simple binary flag.',
    icon: BarChart3,
    accent: 'bg-orange-50 text-orange-600 border-orange-200',
  },
  {
    title: 'Multi-Verdict Classification',
    desc: 'Goes beyond just REAL/FAKE — delivers nuanced verdicts: Verified, Verified Fact, Likely Real, Unverified, Suspicious, or Likely Fake with confidence scores.',
    icon: ShieldAlert,
    accent: 'bg-indigo-50 text-indigo-600 border-indigo-200',
  },
  {
    title: 'Source-Backed Evidence',
    desc: 'Every verdict is backed by clickable source articles with similarity scores — so you can verify the verification yourself. Full transparency, no black boxes.',
    icon: Globe,
    accent: 'bg-teal-50 text-teal-600 border-teal-200',
  },
];

const Features = () => {
  const [headerRef, headerInView] = useInView();

  return (
    <section id="features" className="py-20 px-6 max-w-5xl mx-auto border-t border-black/5">
      {/* Header */}
      <div
        ref={headerRef}
        className={`text-center mb-14 transition-all duration-700 ease-out ${
          headerInView ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
        }`}
      >
        <div className="inline-flex items-center gap-2 px-3 py-1.5 mb-5 rounded-full bg-zinc-100 border border-zinc-200 text-xs font-semibold tracking-wide text-zinc-600 shadow-sm">
          <Sparkles className="w-3.5 h-3.5" />
          Capabilities
        </div>
        <h2 className="text-3xl md:text-4xl font-black tracking-tight text-zinc-900">
          System Features
        </h2>
      </div>

      {/* Feature List */}
      <div className="flex flex-col gap-3">
        {featuresList.map((feature, i) => {
          const [ref, inView] = useInView();
          const Icon = feature.icon;

          return (
            <div
              key={i}
              ref={ref}
              className={`flex items-start gap-4 p-5 rounded-2xl border border-transparent transition-all duration-500 ease-out
                hover:bg-white/60 hover:border-black/5 hover:shadow-md hover:-translate-y-0.5
                ${inView ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-6'}`}
              style={{ transitionDelay: `${i * 60}ms` }}
            >
              {/* Icon */}
              <div className={`mt-0.5 p-2.5 rounded-xl border shadow-sm flex-shrink-0 transition-transform duration-300 hover:scale-110 ${feature.accent}`}>
                <Icon className="w-5 h-5" />
              </div>

              {/* Text */}
              <div>
                <h4 className="text-base font-bold text-zinc-900 mb-1">{feature.title}</h4>
                <p className="text-sm text-zinc-500 leading-relaxed">{feature.desc}</p>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
};

export default Features;