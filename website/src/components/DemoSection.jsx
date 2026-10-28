import React, { useCallback, useEffect, useId, useRef, useState } from 'react';
import { ArrowRight, Loader2, Sparkles, WifiOff, Zap, BookOpen, Newspaper, AlertCircle, RotateCcw, X } from 'lucide-react';
import { motion } from 'framer-motion';
import ChartSection from './analyzer/ChartSection';
import StatsGrid from './analyzer/StatsGrid';
import FeatureBars from './analyzer/FeatureBars';
import VerdictPanel from './analyzer/VerdictPanel';
import InsightsList from './analyzer/InsightsList';
import OnlineVerification from './analyzer/OnlineVerification';
import { fadeUp, staggerContainer } from './analyzer/motion';
import {
  getEnhancedDecisionWithVerification,
  calculateRobustnessWithVerification,
  hasVerificationEvidence,
  isWikiAvailable,
  isWikiContradicted,
  isNewsAvailable,
} from '../utils/decisionLogic';
import { apiUrl, getAnalyzeTimeoutMs, warmUpBackend } from '../utils/api';

const INPUT_TYPE_CONFIG = {
  fact_claim: { label: 'Fact Claim', icon: BookOpen, color: 'text-violet-700 bg-violet-50 border-violet-200' },
  news_article: { label: 'News Article', icon: Newspaper, color: 'text-sky-700 bg-sky-50 border-sky-200' },
  mixed: { label: 'Mixed Content', icon: Zap, color: 'text-amber-700 bg-amber-50 border-amber-200' },
};

const MIN_CHARS = 10;
const MAX_CHARS = 10000;
const REQUEST_TIMEOUT_MS = getAnalyzeTimeoutMs();
// The backend answers within ~8s when awake; longer usually means a cold start
const SLOW_RESPONSE_MS = 10000;

const EXAMPLES = [
  { label: 'Fact claim', text: 'Paris is the capital of France.' },
  { label: 'News headline', text: 'Central bank raises interest rates by a quarter point to curb persistent inflation' },
  { label: 'Clickbait', text: "SHOCKING!!! Doctors HATE this one weird trick that cures everything — you won't BELIEVE what happens next!!!" },
];

const UNKNOWN_MODEL = { label: 'UNKNOWN', confidence: 0.5 };

const normalizeModel = (model) => {
  if (!model || typeof model !== 'object') return { ...UNKNOWN_MODEL };
  const confidence = Number(model.confidence);
  return {
    ...model,
    label: model.label ?? UNKNOWN_MODEL.label,
    confidence: Number.isFinite(confidence) ? confidence : UNKNOWN_MODEL.confidence,
  };
};

const toNumber = (value, fallback) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};

const buildCopySummary = (result) => {
  const typeLabel = (INPUT_TYPE_CONFIG[result.inputType] || INPUT_TYPE_CONFIG.mixed).label;
  const lines = [
    `Verdict: ${result.finalVerdictLabel}`,
    result.confidence != null ? `Confidence: ${Math.round(result.confidence * 100)}%` : null,
    `Input type: ${typeLabel}`,
    `Transformer: ${result.transformer.label} (${Math.round(result.transformer.confidence)}%)`,
    `Hybrid ML: ${result.hybrid.label} (${Math.round(result.hybrid.confidence)}%)`,
    `Model agreement: ${result.agreementScore}%`,
    `Robustness: ${result.robustnessScore}/10`,
    `Clickbait: ${result.features.clickbait}%`,
  ];

  if (result.wikipedia) {
    lines.push(`Wikipedia: ${result.wikipedia.status ?? 'N/A'} (${result.wikipedia.verification_score ?? 0}%)`);
  }
  const articles = result.verification?.articles ?? [];
  if (result.wikipedia?.wiki_url || articles.length > 0) {
    lines.push('Top sources:');
    if (result.wikipedia?.wiki_url) lines.push(`- Wikipedia: ${result.wikipedia.wiki_url}`);
    articles.slice(0, 3).forEach((a) => {
      lines.push(`- ${a.source ?? 'Source'}: ${a.title ?? ''}${a.url ? ` (${a.url})` : ''}`);
    });
  }
  return lines.filter(Boolean).join('\n');
};

const DemoSection = () => {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [loadingPhase, setLoadingPhase] = useState('');
  const [error, setError] = useState(null);

  const abortRef = useRef(null);
  const requestIdRef = useRef(0);
  const verdictHeadingRef = useRef(null);
  const resultsRef = useRef(null);
  const shouldFocusResultRef = useRef(false);

  const textareaId = useId();
  const helperId = useId();

  const trimmed = input.trim();
  const charCount = input.length;
  const wordCount = trimmed ? trimmed.split(/\s+/).length : 0;
  const tooShort = trimmed.length > 0 && trimmed.length < MIN_CHARS;
  const tooLong = charCount > MAX_CHARS;
  const isValid = trimmed.length >= MIN_CHARS && !tooLong;

  // Abort any in-flight request on unmount
  useEffect(() => () => abortRef.current?.abort(), []);

  // Start waking a sleeping backend while the user is still typing
  useEffect(() => {
    warmUpBackend();
  }, []);

  // Move focus to the verdict once a fresh result has rendered
  useEffect(() => {
    if (!result || !shouldFocusResultRef.current) return;
    shouldFocusResultRef.current = false;
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    resultsRef.current?.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
    verdictHeadingRef.current?.focus({ preventScroll: true });
  }, [result]);

  const handleCancel = useCallback(() => {
    abortRef.current?.abort('cancelled');
  }, []);

  const handleAnalyze = async () => {
    if (!isValid) return;

    // Abort a previous in-flight request; only the latest response is applied
    abortRef.current?.abort('superseded');
    const controller = new AbortController();
    abortRef.current = controller;
    const requestId = ++requestIdRef.current;
    const isCurrent = () => requestIdRef.current === requestId;

    let timedOut = false;
    const timeoutId = setTimeout(() => {
      timedOut = true;
      controller.abort('timeout');
    }, REQUEST_TIMEOUT_MS);
    const slowId = setTimeout(() => {
      if (isCurrent()) {
        setLoadingPhase('Still working — the server may be waking up. The first request can take up to a minute.');
      }
    }, SLOW_RESPONSE_MS);

    const text = trimmed;
    setLoading(true);
    setError(null);
    setLoadingPhase('Running 6 parallel tasks...');

    try {
      let response;
      try {
        response = await fetch(apiUrl('/analyze'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
          signal: controller.signal,
        });
      } catch (networkErr) {
        if (controller.signal.aborted) throw networkErr;
        const err = new Error('Could not reach the analysis server. Check that the backend is running and reachable, then try again.');
        err.cause = networkErr;
        throw err;
      }

      if (!response.ok) {
        let detail = '';
        try {
          const body = await response.json();
          detail = body?.error ? ` ${body.error}` : '';
        } catch {
          /* non-JSON error body */
        }
        throw new Error(`The server responded with HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ''}.${detail}`);
      }

      let apiData;
      try {
        apiData = await response.json();
      } catch {
        throw new Error('The server returned a response that could not be read.');
      }
      if (!isCurrent()) return;

      if (!apiData || typeof apiData !== 'object') {
        throw new Error('The server returned an empty response.');
      }
      if (apiData.error) {
        throw new Error(`Analysis failed: ${apiData.error}`);
      }

      setLoadingPhase('Computing verdict...');

      const words = text.split(/\s+/);
      const inputWordCount = words.length;

      // Extract from unified response (tolerate partial payloads)
      const transformer = normalizeModel(apiData.transformer);
      const hybrid = normalizeModel(apiData.hybrid);
      const t_conf = transformer.confidence;
      const t_label = transformer.label;
      const h_conf = hybrid.confidence;
      const h_label = hybrid.label;
      const rawFeatures = apiData.raw_features ?? {};
      const inputType = apiData.input_type ?? 'mixed';
      const wikipedia = apiData.wikipedia ?? null;
      const verification = apiData.verification ?? null;

      const computedFeatures = {
        clickbait: toNumber(rawFeatures.clickbait, 0),
        uppercase: toNumber(rawFeatures.uppercase, 0),
        punctuation: toNumber(rawFeatures.punctuation, 0),
        complexity: toNumber(rawFeatures.complexity, 5),
      };

      // Scale features for decision logic
      const clickbaitScore = Math.round(computedFeatures.clickbait * 100);
      const uppercaseScore = Math.round(computedFeatures.uppercase * 100);
      const pctScore = Math.round(computedFeatures.punctuation * 100);

      const hasVerification = hasVerificationEvidence(wikipedia, verification);

      // Always use the verification-aware decision: with no online evidence it
      // returns UNVERIFIED (noting the ML lean) instead of an ML-only REAL/FAKE.
      const decision = getEnhancedDecisionWithVerification(
        t_label, t_conf,
        h_label, h_conf,
        clickbaitScore, pctScore, uppercaseScore,
        inputWordCount,
        inputType,
        wikipedia,
        verification
      );
      const robustnessScore = calculateRobustnessWithVerification(
        t_conf, h_conf, decision.agreement, inputWordCount, wikipedia, verification
      );

      const dashboardData = {
        transformer: {
          label: t_label,
          confidence: t_conf * 100,
          source: transformer.source ?? 'local'
        },
        hybrid: {
          label: h_label,
          confidence: h_conf * 100
        },
        stats: {
          wordCount: inputWordCount,
          avgWordLength: (text.length / Math.max(1, inputWordCount)).toFixed(1),
          upperRatio: uppercaseScore,
          punctDensity: pctScore
        },
        features: {
          punctuation: pctScore,
          uppercase: uppercaseScore,
          complexity: computedFeatures.complexity,
          clickbait: clickbaitScore
        },
        confidence: decision.confidence ?? null,
        agreementScore: decision.agreement,
        robustnessScore: robustnessScore,
        finalVerdictLabel: decision.final_label,
        verdictExplanation: decision.reason,
        explanations: decision.insights ?? (decision.reason ? [decision.reason] : []),
        inputType: inputType,
        // Keep contradicting Wikipedia evidence visible, not only supporting matches
        wikipedia: isWikiAvailable(wikipedia) || isWikiContradicted(wikipedia) ? wikipedia : null,
        verification: isNewsAvailable(verification) ? verification : null,
        verificationSource: decision.verification_source ?? 'none',
        verificationWeights: decision.weights ?? null,
        verificationFailed: !hasVerification,
        timings: apiData.timings ?? null,
        totalTime: apiData.total_time ?? null,
        // Display-only: the submitted text, shown as a quoted excerpt in the verdict card
        analyzedText: text
      };

      if (!isCurrent()) return;
      shouldFocusResultRef.current = true;
      setResult(dashboardData);
    } catch (err) {
      if (!isCurrent()) return;
      if (controller.signal.aborted) {
        if (timedOut) {
          setError(`The analysis took longer than ${Math.round(REQUEST_TIMEOUT_MS / 1000)} seconds and was stopped. The server may still be starting up — please try again in a moment.`);
        }
        // User-cancelled: no error message
      } else {
        console.error(err);
        setError(err?.message || 'Something went wrong while analysing the text.');
      }
    } finally {
      clearTimeout(timeoutId);
      clearTimeout(slowId);
      if (isCurrent()) {
        setLoading(false);
        setLoadingPhase('');
        abortRef.current = null;
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (!loading) handleAnalyze();
    }
  };

  const handleClear = () => {
    setInput('');
    setError(null);
  };

  const fillExample = (text) => {
    setInput(text);
    setError(null);
  };

  let helperText = `Minimum ${MIN_CHARS} characters. Press Ctrl/⌘ + Enter to analyze.`;
  if (tooShort) helperText = `Enter at least ${MIN_CHARS} characters (${MIN_CHARS - trimmed.length} more needed).`;
  if (tooLong) helperText = `Text is too long — maximum ${MAX_CHARS.toLocaleString()} characters (${(charCount - MAX_CHARS).toLocaleString()} over).`;
  const helperIsError = tooShort || tooLong;

  return (
    <section id="demo" className="py-10 px-6 max-w-6xl mx-auto">
      <div className="text-center mb-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 mb-6 rounded-full bg-zinc-100 border border-zinc-200 text-xs font-semibold tracking-wide text-zinc-600 shadow-sm">
          <Sparkles className="w-4 h-4" aria-hidden="true" />
          Intelligence Dashboard
        </div>
        <h2 className="text-3xl md:text-5xl font-black tracking-tight mb-4 text-zinc-900">
          Content Analysis Engine
        </h2>
        <p className="text-zinc-500 max-w-xl mx-auto text-sm md:text-base font-medium">
          Paste your article or headline below. Our parallel verification engine runs 6 concurrent checks — ML models, linguistic analysis, Wikipedia facts, and live news sources.
        </p>
      </div>

      <div className="glass-card p-6 md:p-8 rounded-3xl bg-white/70 shadow-xl border border-black/5 ring-1 ring-black/5 mb-12 mx-auto max-w-4xl relative z-10 transition-all">
        {/* Example chips */}
        <div className="flex flex-wrap items-center gap-2 mb-4">
          <span className="text-xs font-semibold text-zinc-500">Try an example:</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex.label}
              type="button"
              onClick={() => fillExample(ex.text)}
              disabled={loading}
              className="px-3 py-1.5 rounded-full bg-zinc-100 border border-zinc-200 text-xs font-semibold text-zinc-700 hover:bg-zinc-200 transition-colors disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900/40"
            >
              {ex.label}
            </button>
          ))}
        </div>

        <div className="relative mb-2">
          <label htmlFor={textareaId} className="sr-only">
            Article or headline text to analyze
          </label>
          <textarea
            id={textareaId}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Paste text here... "
            aria-describedby={helperId}
            aria-invalid={helperIsError || undefined}
            className={`w-full min-h-[160px] p-6 rounded-2xl bg-white/80 border text-zinc-900 placeholder:text-zinc-500 focus:outline-none focus:ring-4 focus:ring-zinc-900/5 focus:bg-white transition-all resize-y text-base md:text-lg font-medium shadow-inner ${
              helperIsError ? 'border-red-300 focus:border-red-400' : 'border-zinc-200 focus:border-zinc-300'
            }`}
          />
        </div>

        <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-1 mb-6 px-1">
          <p id={helperId} className={`text-xs font-medium ${helperIsError ? 'text-red-700' : 'text-zinc-500'}`}>
            {helperText}
          </p>
          <p className={`text-xs font-semibold tabular-nums ${tooLong ? 'text-red-700' : 'text-zinc-500'}`} aria-live="off">
            {wordCount.toLocaleString()} {wordCount === 1 ? 'word' : 'words'} · {charCount.toLocaleString()} / {MAX_CHARS.toLocaleString()} chars
          </p>
        </div>

        {error && (
          <div
            role="alert"
            className="flex flex-col sm:flex-row sm:items-center gap-3 mb-6 px-4 py-3 rounded-xl bg-red-50 border border-red-200 text-red-800"
          >
            <div className="flex items-start gap-2 flex-1">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" aria-hidden="true" />
              <p className="text-sm font-medium">{error}</p>
            </div>
            <button
              type="button"
              onClick={handleAnalyze}
              disabled={!isValid || loading}
              className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-red-200 text-xs font-bold text-red-800 hover:bg-red-100 transition-colors disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
            >
              <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
              Retry
            </button>
          </div>
        )}

        <div className="flex flex-col-reverse md:flex-row md:justify-end gap-3">
          {loading ? (
            <button
              type="button"
              onClick={handleCancel}
              className="w-full md:w-auto px-6 py-4 rounded-xl border border-zinc-200 bg-white text-zinc-700 font-bold tracking-wide hover:bg-zinc-50 transition-colors flex items-center justify-center gap-2 focus:outline-none focus-visible:ring-4 focus-visible:ring-zinc-900/20"
            >
              <X className="w-4 h-4" aria-hidden="true" />
              Cancel
            </button>
          ) : (
            input.length > 0 && (
              <button
                type="button"
                onClick={handleClear}
                className="w-full md:w-auto px-6 py-4 rounded-xl border border-zinc-200 bg-white text-zinc-700 font-bold tracking-wide hover:bg-zinc-50 transition-colors flex items-center justify-center gap-2 focus:outline-none focus-visible:ring-4 focus-visible:ring-zinc-900/20"
              >
                Clear
              </button>
            )
          )}
          <button
            type="button"
            onClick={handleAnalyze}
            disabled={loading || !isValid}
            className="w-full md:w-auto px-10 py-4 bg-zinc-900 text-white rounded-xl font-bold tracking-wide hover:bg-zinc-800 focus:outline-none focus-visible:ring-4 focus-visible:ring-zinc-900/30 transition-all shadow-lg hover:-translate-y-0.5 disabled:opacity-50 disabled:hover:translate-y-0 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" aria-hidden="true" />
                {loadingPhase || 'Analyzing...'}
              </>
            ) : (
              <>
                Initialize Analysis
                <ArrowRight className="w-4 h-4 ml-1" aria-hidden="true" />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Analytics Dashboard */}
      <div
        ref={resultsRef}
        aria-live="polite"
        aria-busy={loading}
        className="scroll-mt-24"
      >
        {loading && !result && <ResultsSkeleton />}

        {result && (
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            animate="show"
            className={`space-y-4 transition-opacity duration-300 sm:space-y-6 ${
              loading ? 'opacity-50 pointer-events-none select-none' : 'opacity-100'
            }`}
          >
            {/* Hero verdict + evidence breakdown */}
            <VerdictPanel
              label={result.finalVerdictLabel}
              transformerLabel={result.transformer.label}
              hybridLabel={result.hybrid.label}
              explanation={result.verdictExplanation}
              agreementScore={result.agreementScore}
              robustnessScore={result.robustnessScore}
              verificationWeights={result.verificationWeights}
              inputType={result.inputType}
              headingRef={verdictHeadingRef}
              copyText={buildCopySummary(result)}
              confidence={result.confidence}
              totalTime={result.totalTime}
              excerpt={result.analyzedText}
            />

            {/* Verification status banner */}
            {result.verificationFailed && (
              <motion.div
                variants={fadeUp}
                role="note"
                className="flex items-start gap-3 rounded-2xl border border-amber-200/80 bg-amber-50/70 px-4 py-3 text-amber-900"
              >
                <span className="mt-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-lg bg-amber-100 text-amber-700">
                  <WifiOff className="h-3.5 w-3.5" aria-hidden="true" />
                </span>
                <p className="text-[13px] leading-5">
                  <span className="font-semibold">Online verification unavailable.</span>{' '}
                  <span className="text-amber-800">The verdict is based on ML models and linguistic analysis only.</span>
                </p>
              </motion.div>
            )}

            {/* Bento grid: 1 col on mobile, 2 on md, 3 on lg (verification spans two rows) */}
            <motion.div
              variants={staggerContainer}
              className="grid grid-cols-1 gap-4 sm:gap-6 md:grid-cols-2 lg:grid-cols-3"
            >
              <ChartSection
                transformerConf={result.transformer.confidence}
                hybridConf={result.hybrid.confidence}
                transformerLabel={result.transformer.label}
                hybridLabel={result.hybrid.label}
              />
              <FeatureBars {...result.features} />
              <div className="min-w-0 md:col-span-2 lg:col-span-1 lg:row-span-2">
                <OnlineVerification
                  wikipedia={result.wikipedia}
                  verification={result.verification}
                  verificationSource={result.verificationSource}
                  inputType={result.inputType}
                />
              </div>
              <div className="min-w-0 md:col-span-2">
                <StatsGrid {...result.stats} />
              </div>
              <div className="min-w-0 md:col-span-2 lg:col-span-3">
                <InsightsList explanations={result.explanations} />
              </div>
            </motion.div>
          </motion.div>
        )}
      </div>
    </section>
  );
};

const SkeletonBlock = ({ className = '' }) => (
  <div className={`rounded-2xl bg-zinc-200/50 animate-pulse ${className}`} />
);

const ResultsSkeleton = () => (
  <div className="space-y-4 sm:space-y-6" aria-hidden="true">
    {/* Hero */}
    <div className="overflow-hidden rounded-3xl border border-zinc-200/70 bg-white shadow-card">
      <div className="space-y-6 p-6 sm:p-8">
        <div className="flex items-center justify-between gap-3">
          <SkeletonBlock className="h-8 w-40 rounded-full" />
          <SkeletonBlock className="h-10 w-28 rounded-xl" />
        </div>
        <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex-1 space-y-3">
            <SkeletonBlock className="h-3 w-24 rounded-md" />
            <SkeletonBlock className="h-12 w-64 max-w-full rounded-xl" />
            <SkeletonBlock className="h-4 w-full max-w-lg rounded-md" />
            <SkeletonBlock className="h-4 w-4/5 max-w-md rounded-md" />
          </div>
          <SkeletonBlock className="h-24 w-24 rounded-full sm:h-28 sm:w-28 lg:h-44 lg:w-44" />
        </div>
      </div>
      <div className="space-y-3 border-t border-zinc-200/70 bg-zinc-50/50 p-6 sm:p-8">
        <SkeletonBlock className="h-3 w-32 rounded-md" />
        <SkeletonBlock className="h-2.5 w-full rounded-full" />
      </div>
    </div>
    {/* Bento */}
    <div className="grid grid-cols-1 gap-4 sm:gap-6 md:grid-cols-2 lg:grid-cols-3">
      <SkeletonBlock className="h-56" />
      <SkeletonBlock className="h-56" />
      <SkeletonBlock className="h-56 md:col-span-2 lg:col-span-1 lg:row-span-2 lg:h-auto" />
      <SkeletonBlock className="h-36 md:col-span-2" />
    </div>
  </div>
);

export default DemoSection;
