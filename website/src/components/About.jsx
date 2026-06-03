import React, { useEffect, useRef, useState } from 'react';
import {
  BrainCircuit, BookOpen, Newspaper, Globe, Cpu,
  Search, ShieldCheck, Zap, Layers, ArrowRight,
  Fingerprint, BarChart3
} from 'lucide-react';

/* ── Intersection Observer hook for scroll-triggered animations ── */
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

/* ── Animated counter ── */
const AnimatedNumber = ({ target, suffix = '', duration = 1400 }) => {
  const [value, setValue] = useState(0);
  const [ref, isInView] = useInView();

  useEffect(() => {
    if (!isInView) return;
    let start = 0;
    const step = Math.ceil(target / (duration / 16));
    const timer = setInterval(() => {
      start += step;
      if (start >= target) { setValue(target); clearInterval(timer); }
      else setValue(start);
    }, 16);
    return () => clearInterval(timer);
  }, [isInView, target, duration]);

  return <span ref={ref}>{value}{suffix}</span>;
};

/* ── Pipeline steps data ── */
const PIPELINE = [
  {
    icon: BrainCircuit,
    title: 'Transformer Model',
    tag: 'Deep Learning',
    tagColor: 'text-violet-700 bg-violet-50 border-violet-200',
    iconBg: 'bg-violet-50',
    iconColor: 'text-violet-600',
    description: 'Fine-tuned DistilBERT neural network analyzes deep semantic meaning and context of statements across thousands of parameters.',
    tech: ['DistilBERT', 'HuggingFace', 'PyTorch'],
  },
  {
    icon: Cpu,
    title: 'LR + TF-IDF Model',
    tag: 'Machine Learning',
    tagColor: 'text-sky-700 bg-sky-50 border-sky-200',
    iconBg: 'bg-sky-50',
    iconColor: 'text-sky-600',
    description: 'Logistic Regression on TF-IDF features acts as a fast, interpretable second opinion — catching patterns the transformer may miss.',
    tech: ['Scikit-learn', 'TF-IDF', 'Logistic Regression'],
  },
  {
    icon: Fingerprint,
    title: 'Linguistic Analysis',
    tag: 'NLP Features',
    tagColor: 'text-amber-700 bg-amber-50 border-amber-200',
    iconBg: 'bg-amber-50',
    iconColor: 'text-amber-600',
    description: '8 handcrafted features — clickbait detection, punctuation density, uppercase ratio, text complexity — flag sensationalist writing patterns.',
    tech: ['Clickbait Score', 'Stylometry', 'Heuristics'],
  },
  {
    icon: Newspaper,
    title: 'Multi-Source News Verification',
    tag: '4 News APIs',
    tagColor: 'text-emerald-700 bg-emerald-50 border-emerald-200',
    iconBg: 'bg-emerald-50',
    iconColor: 'text-emerald-600',
    description: 'Queries GNews, NewsData.io, NewsAPI.org in parallel and uses SerpAPI (Google News) as a smart fallback when results are insufficient.',
    tech: ['GNews', 'NewsData.io', 'NewsAPI', 'SerpAPI'],
  },
  {
    icon: BookOpen,
    title: 'Wikipedia Verification',
    tag: 'Knowledge Base',
    tagColor: 'text-indigo-700 bg-indigo-50 border-indigo-200',
    iconBg: 'bg-indigo-50',
    iconColor: 'text-indigo-600',
    description: 'Extracts key entities and verifies factual claims against Wikipedia using TF-IDF cosine similarity and contradiction detection.',
    tech: ['Wikipedia REST API', 'Cosine Similarity', 'Entity Extraction'],
  },
  {
    icon: BarChart3,
    title: 'Weighted Verdict Engine',
    tag: 'Decision Logic',
    tagColor: 'text-rose-700 bg-rose-50 border-rose-200',
    iconBg: 'bg-rose-50',
    iconColor: 'text-rose-600',
    description: 'Input-aware weighting dynamically adjusts component influence — news articles prioritize live sources, fact claims prioritize Wikipedia.',
    tech: ['Input Classification', 'Dynamic Weights', 'Override Rules'],
  },
];

/* ── Stats data ── */
const STATS = [
  { label: 'Parallel Tasks', value: 6, suffix: '' },
  { label: 'News APIs', value: 4, suffix: '+' },
  { label: 'ML Features', value: 8, suffix: '' },
  { label: 'Trusted Sources', value: 10, suffix: '+' },
];

const About = () => {
  const [headerRef, headerInView] = useInView();
  const [statsRef, statsInView] = useInView();
  const [archRef, archInView] = useInView();

  return (
    <section id="about" className="py-20 px-6 max-w-6xl mx-auto">

      {/* ── Section Header ── */}
      <div
        ref={headerRef}
        className={`text-center mb-16 transition-all duration-700 ease-out ${
          headerInView ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
        }`}
      >
        <div className="inline-flex items-center gap-2 px-3 py-1.5 mb-6 rounded-full bg-zinc-100 border border-zinc-200 text-xs font-semibold tracking-wide text-zinc-600 shadow-sm">
          <Layers className="w-4 h-4" />
          System Architecture
        </div>
        <h2 className="text-3xl md:text-5xl font-black tracking-tight mb-4 text-zinc-900">
          How It Works
        </h2>
        <p className="text-zinc-500 max-w-2xl mx-auto text-sm md:text-base font-medium leading-relaxed">
          Six concurrent verification tasks run in parallel — combining deep learning, classical ML,
          linguistic analysis, and live cross-referencing across multiple news APIs and Wikipedia.
        </p>
      </div>

      {/* ── Stats Row ── */}
      <div
        ref={statsRef}
        className={`grid grid-cols-2 md:grid-cols-4 gap-4 mb-16 transition-all duration-700 ease-out delay-150 ${
          statsInView ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
        }`}
      >
        {STATS.map((stat, i) => (
          <div
            key={i}
            className="glass-card p-6 text-center group hover:bg-white/80 transition-all duration-300 hover:-translate-y-1 hover:shadow-lg"
            style={{ transitionDelay: `${i * 80}ms` }}
          >
            <div className="text-3xl md:text-4xl font-black text-zinc-900 mb-1">
              <AnimatedNumber target={stat.value} suffix={stat.suffix} />
            </div>
            <div className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
              {stat.label}
            </div>
          </div>
        ))}
      </div>

      {/* ── Architecture Label ── */}
      <div
        ref={archRef}
        className={`flex items-center gap-3 mb-8 transition-all duration-500 ease-out ${
          archInView ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-6'
        }`}
      >
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900 text-white text-xs font-bold uppercase tracking-wider">
          <Zap className="w-3.5 h-3.5" />
          Pipeline
        </div>
        <div className="h-px flex-1 bg-zinc-200" />
        <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
          All tasks execute concurrently
        </span>
      </div>

      {/* ── Pipeline Cards ── */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5 mb-12">
        {PIPELINE.map((step, i) => {
          const [cardRef, cardInView] = useInView();
          const Icon = step.icon;

          return (
            <div
              key={i}
              ref={cardRef}
              className={`glass-card p-6 group hover:bg-white/80 transition-all duration-500 ease-out hover:-translate-y-1.5 hover:shadow-xl ${
                cardInView ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
              }`}
              style={{ transitionDelay: `${i * 100}ms` }}
            >
              {/* Icon + Tag */}
              <div className="flex items-start justify-between mb-5">
                <div className={`w-11 h-11 ${step.iconBg} rounded-xl flex items-center justify-center transition-transform duration-300 group-hover:scale-110 group-hover:rotate-3`}>
                  <Icon className={`w-5.5 h-5.5 ${step.iconColor}`} />
                </div>
                <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded-md border ${step.tagColor}`}>
                  {step.tag}
                </span>
              </div>

              {/* Title */}
              <h3 className="text-lg font-bold text-zinc-900 mb-2 group-hover:text-zinc-800 transition-colors">
                {step.title}
              </h3>

              {/* Description */}
              <p className="text-zinc-500 text-sm leading-relaxed mb-4">
                {step.description}
              </p>

              {/* Tech pills */}
              <div className="flex flex-wrap gap-1.5">
                {step.tech.map((t, j) => (
                  <span
                    key={j}
                    className="text-[10px] font-semibold px-2 py-0.5 rounded-md bg-zinc-100 text-zinc-500 border border-zinc-200/50 transition-colors group-hover:bg-zinc-50 group-hover:text-zinc-600"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Flow Diagram ── */}
      <FlowDiagram />

    </section>
  );
};


/* ── Visual Flow Diagram ── */
const FlowDiagram = () => {
  const [ref, isInView] = useInView();

  const steps = [
    { icon: Search, label: 'Input', sub: 'Classify' },
    { icon: Zap, label: '6 Tasks', sub: 'Parallel' },
    { icon: Globe, label: 'Verify', sub: 'Cross-ref' },
    { icon: ShieldCheck, label: 'Verdict', sub: 'Weighted' },
  ];

  return (
    <div
      ref={ref}
      className={`glass-card p-6 md:p-8 transition-all duration-700 ease-out ${
        isInView ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
      }`}
    >
      <div className="flex items-center gap-2 mb-6">
        <span className="text-xs font-bold text-zinc-900 uppercase tracking-wider">Analysis Flow</span>
        <div className="h-px flex-1 bg-zinc-200" />
      </div>

      <div className="flex items-center justify-between max-w-2xl mx-auto">
        {steps.map((step, i) => {
          const Icon = step.icon;
          return (
            <React.Fragment key={i}>
              <div
                className={`flex flex-col items-center gap-2 transition-all duration-500 ease-out ${
                  isInView ? 'opacity-100 scale-100' : 'opacity-0 scale-75'
                }`}
                style={{ transitionDelay: `${300 + i * 200}ms` }}
              >
                <div className="w-12 h-12 md:w-14 md:h-14 rounded-2xl bg-zinc-900 flex items-center justify-center shadow-lg shadow-zinc-900/20 transition-transform duration-300 hover:scale-110">
                  <Icon className="w-5 h-5 md:w-6 md:h-6 text-white" />
                </div>
                <span className="text-xs font-bold text-zinc-800">{step.label}</span>
                <span className="text-[10px] font-medium text-zinc-400">{step.sub}</span>
              </div>

              {i < steps.length - 1 && (
                <div
                  className={`flex-1 flex items-center justify-center transition-all duration-500 ease-out ${
                    isInView ? 'opacity-100' : 'opacity-0'
                  }`}
                  style={{ transitionDelay: `${500 + i * 200}ms` }}
                >
                  <div className="h-px flex-1 bg-zinc-300 mx-1" />
                  <ArrowRight className="w-3.5 h-3.5 text-zinc-400 flex-shrink-0" />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>

      <p className="text-center text-xs text-zinc-400 font-medium mt-6">
        Average analysis completes in <span className="font-bold text-zinc-600">~5 seconds</span> — all 6 tasks run concurrently via ThreadPoolExecutor
      </p>
    </div>
  );
};

export default About;