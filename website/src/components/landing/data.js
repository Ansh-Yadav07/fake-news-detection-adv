import {
  BookOpen,
  BrainCircuit,
  Cpu,
  Fingerprint,
  Newspaper,
  ScanSearch,
  Layers,
  ShieldCheck,
} from 'lucide-react';

// Signal colours match analyzer/EvidenceBreakdown.jsx so the landing page and
// the results view speak the same visual language.
export const SIGNAL_COLORS = {
  wikipedia: '#8b5cf6', // violet-500
  news: '#0ea5e9', // sky-500
  ml: '#3f3f46', // zinc-700
  credibility: '#10b981', // emerald-500
  linguistic: '#f59e0b', // amber-500
  clickbait: '#fb7185', // rose-400
};

export const SIGNAL_LABELS = {
  wikipedia: 'Wikipedia',
  news: 'News',
  ml: 'ML models',
  credibility: 'Source credibility',
  linguistic: 'Linguistic',
  clickbait: 'Clickbait',
};

// Mirrors FACT_WEIGHTS / NEWS_WEIGHTS / MIXED_WEIGHTS in src/utils/decisionLogic.js
// (base profiles; renormalised over available signals at runtime).
export const WEIGHT_PROFILES = [
  {
    id: 'fact',
    label: 'Fact claim',
    Icon: BookOpen,
    weights: [
      ['wikipedia', 60],
      ['ml', 20],
      ['linguistic', 10],
      ['clickbait', 10],
    ],
  },
  {
    id: 'news',
    label: 'News article',
    Icon: Newspaper,
    weights: [
      ['news', 50],
      ['ml', 20],
      ['credibility', 15],
      ['linguistic', 10],
      ['clickbait', 5],
    ],
  },
  {
    id: 'mixed',
    label: 'Mixed content',
    Icon: Layers,
    weights: [
      ['wikipedia', 30],
      ['news', 25],
      ['ml', 20],
      ['linguistic', 15],
      ['clickbait', 10],
    ],
  },
];

// The five evidence checks (the sixth parallel task is the verdict engine,
// rendered as the featured card in Features.jsx).
export const CHECKS = [
  {
    title: 'Transformer model',
    desc: 'A fine-tuned DistilBERT network reads semantic meaning and context, with a hosted Hugging Face fallback.',
    Icon: BrainCircuit,
    tile: 'bg-violet-50 text-violet-700 ring-violet-600/15',
    tags: ['DistilBERT', 'Deep learning'],
  },
  {
    title: 'LR + TF-IDF model',
    desc: 'Logistic regression on TF-IDF features: a fast, interpretable second opinion that catches what the transformer misses.',
    Icon: Cpu,
    tile: 'bg-zinc-100 text-zinc-700 ring-zinc-500/15',
    tags: ['scikit-learn', 'Classical ML'],
  },
  {
    title: 'Linguistic analysis',
    desc: '8 handcrafted style features such as punctuation density, uppercase ratio and complexity, plus a 0–100% clickbait score.',
    Icon: Fingerprint,
    tile: 'bg-amber-50 text-amber-700 ring-amber-600/15',
    tags: ['8 features', 'Clickbait'],
  },
  {
    title: 'Wikipedia verification',
    desc: 'Extracts entities and matches claims by TF-IDF cosine similarity, and flags contradictions like a wrong capital city.',
    Icon: BookOpen,
    tile: 'bg-indigo-50 text-indigo-700 ring-indigo-600/15',
    tags: ['Contradiction detection', 'REST API'],
  },
  {
    title: 'Multi-source news',
    desc: 'Queries GNews, NewsData.io and NewsAPI.org in parallel, falling back to SerpAPI (Google News) when results are thin.',
    Icon: Newspaper,
    tile: 'bg-sky-50 text-sky-700 ring-sky-600/15',
    tags: ['4 news APIs', 'Trusted outlets'],
  },
];

export const STEPS = [
  {
    title: 'Input classification',
    desc: 'Your text is classified as a fact claim, a news article or mixed content, which picks the weight profile.',
    Icon: ScanSearch,
  },
  {
    title: '6 parallel checks',
    desc: 'Models, linguistic features, Wikipedia and news lookups run concurrently within an 8-second default budget.',
    Icon: Cpu,
  },
  {
    title: 'Cross-verification',
    desc: 'Claims are matched against Wikipedia and live coverage, with priority for trusted outlets such as Reuters, AP and BBC.',
    Icon: Newspaper,
  },
  {
    title: 'Weighted verdict',
    desc: 'Signals are combined with input-aware weights and override rules into a verdict, a confidence and the evidence behind it.',
    Icon: ShieldCheck,
  },
];

// Verdict labels decisionLogic.js can return, grouped by meaning. Display
// text, taglines and tones come from analyzer/verdictTheme.js.
export const VERDICT_GROUPS = [
  { id: 'supported', title: 'Supported', keys: ['VERIFIED FACT', 'VERIFIED', 'LIKELY REAL', 'REAL'] },
  { id: 'inconclusive', title: 'Inconclusive', keys: ['UNVERIFIED', 'UNCERTAIN'] },
  { id: 'warning', title: 'Warning signs', keys: ['SUSPICIOUS', 'LIKELY FAKE', 'FAKE'] },
];

export const BUILT_ON = ['DistilBERT', 'scikit-learn', 'Wikipedia', 'GNews', 'NewsData.io', 'NewsAPI.org', 'SerpAPI'];

