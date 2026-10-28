import React, { useEffect, useId, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { BookOpen, Check, Copy, Layers, Newspaper, Timer, X } from 'lucide-react';
import Ring from './Ring';
import EvidenceBreakdown from './EvidenceBreakdown';
import { fadeUp } from './motion';
import { resolveVerdict, rgba } from './verdictTheme';

const inputTypeMeta = {
  fact_claim: { label: 'Fact claim', Icon: BookOpen },
  news_article: { label: 'News article', Icon: Newspaper },
  mixed: { label: 'Mixed content', Icon: Layers },
};

const EXCERPT_CLAMP_CHARS = 180;

const copyToClipboard = async (text) => {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  // Fallback for non-secure contexts (e.g. LAN IP over http)
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.select();
  const ok = document.execCommand('copy');
  document.body.removeChild(textarea);
  if (!ok) throw new Error('Copy command was rejected');
};

const formatSeconds = (value) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return `${n < 10 ? n.toFixed(1) : Math.round(n)}s`;
};

const Chip = ({ icon: Icon, children, title }) => (
  <span
    title={title}
    className="inline-flex h-8 items-center gap-1.5 rounded-full border border-zinc-200/70 bg-white/80 px-3 text-xs font-medium text-zinc-600"
  >
    {Icon && <Icon className="h-3.5 w-3.5 text-zinc-400" aria-hidden="true" />}
    {children}
  </span>
);

const CopyButton = ({ text }) => {
  const [copyState, setCopyState] = useState('idle'); // idle | copied | error
  const resetTimer = useRef(null);

  useEffect(() => () => clearTimeout(resetTimer.current), []);

  const handleCopy = async () => {
    try {
      await copyToClipboard(text);
      setCopyState('copied');
    } catch {
      setCopyState('error');
    }
    clearTimeout(resetTimer.current);
    resetTimer.current = setTimeout(() => setCopyState('idle'), 2000);
  };

  const Icon = copyState === 'copied' ? Check : copyState === 'error' ? X : Copy;

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="inline-flex min-h-10 items-center gap-2 rounded-xl border border-zinc-200/70 bg-white px-3.5 text-[13px] font-medium text-zinc-700 shadow-sm transition-colors hover:border-zinc-300 hover:bg-zinc-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900/40 focus-visible:ring-offset-2"
    >
      <Icon className={`h-4 w-4 ${copyState === 'copied' ? 'text-emerald-600' : 'text-zinc-500'}`} aria-hidden="true" />
      <span aria-live="polite">
        {copyState === 'copied' ? 'Copied' : copyState === 'error' ? 'Copy failed' : 'Copy result'}
      </span>
    </button>
  );
};

const Excerpt = ({ text, accentRgb }) => {
  const [expanded, setExpanded] = useState(false);
  const excerptId = useId();
  const clampable = text.length > EXCERPT_CLAMP_CHARS || text.includes('\n');

  return (
    <figure className="mt-6">
      <figcaption className="micro-label mb-2">Analysed text</figcaption>
      <blockquote
        id={excerptId}
        className={`whitespace-pre-line break-words border-l-2 pl-4 text-sm leading-6 text-zinc-600 ${
          clampable && !expanded ? 'line-clamp-3' : ''
        }`}
        style={{ borderColor: rgba(accentRgb, 0.35) }}
      >
        {text}
      </blockquote>
      {clampable && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
          aria-controls={excerptId}
          className="-ml-2 mt-1 inline-flex min-h-10 items-center rounded-lg px-2 text-[13px] font-medium text-zinc-700 underline-offset-4 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900/40"
        >
          {expanded ? 'Show less' : 'Show full text'}
        </button>
      )}
    </figure>
  );
};

const ConfidenceGauge = ({ confidence, accentRgb }) => {
  const raw = Number(confidence);
  const available = confidence != null && Number.isFinite(raw);
  const pct = available ? Math.round(Math.min(1, Math.max(0, raw)) * 100) : null;
  const qualifier = !available ? 'Unavailable' : pct >= 80 ? 'High confidence' : pct >= 60 ? 'Moderate confidence' : 'Low confidence';

  return (
    <div className="flex items-center gap-6 rounded-2xl border border-zinc-200/70 bg-white/70 p-4 lg:flex-col lg:gap-4 lg:border-0 lg:bg-transparent lg:p-0">
      <Ring
        value={pct ?? 0}
        unavailable={!available}
        label={available ? `Verdict confidence ${pct}%` : 'Verdict confidence unavailable'}
        color={rgba(accentRgb, 1)}
        trackColor={rgba(accentRgb, 0.1)}
        strokeWidth={7}
        sizeClass="h-24 w-24 sm:h-28 sm:w-28 lg:h-44 lg:w-44"
      >
        {available ? (
          <span className="flex items-start font-semibold leading-none tracking-[-0.04em] text-zinc-950 tabular-nums">
            <span className="text-[28px] sm:text-[32px] lg:text-[48px]">{pct}</span>
            <span className="mt-0.5 text-sm text-zinc-400 lg:mt-1.5 lg:text-lg">%</span>
          </span>
        ) : (
          <span className="text-2xl font-semibold text-zinc-300 lg:text-4xl">—</span>
        )}
      </Ring>
      <div className="lg:text-center">
        <p className="micro-label">Confidence</p>
        <p className="mt-1 text-sm font-medium text-zinc-700">{qualifier}</p>
      </div>
    </div>
  );
};

const VerdictPanel = ({
  label,
  transformerLabel,
  hybridLabel,
  explanation,
  agreementScore,
  robustnessScore,
  verificationWeights,
  inputType,
  headingRef,
  copyText,
  confidence,
  totalTime,
  excerpt,
}) => {
  const tone = resolveVerdict(label);
  const { Icon } = tone;
  const itMeta = inputTypeMeta[inputType] || inputTypeMeta.mixed;
  const timing = formatSeconds(totalTime);
  const accent = rgba(tone.rgb, 1);

  return (
    <motion.article
      variants={fadeUp}
      className="relative isolate overflow-hidden rounded-3xl border border-zinc-200/70 bg-white shadow-hero"
    >
      {/* Verdict-tinted aurora wash + soft inner ring (decorative) */}
      <div
        className="pointer-events-none absolute inset-0 -z-10"
        aria-hidden="true"
        style={{
          backgroundImage: [
            `radial-gradient(70% 90% at 0% 0%, ${rgba(tone.rgb, 0.11)} 0%, transparent 60%)`,
            `radial-gradient(45% 70% at 100% 0%, ${rgba(tone.rgb, 0.07)} 0%, transparent 70%)`,
          ].join(', '),
          boxShadow: `inset 0 0 0 1px ${rgba(tone.rgb, 0.1)}`,
          borderRadius: 'inherit',
        }}
      />
      <div
        className="pointer-events-none absolute inset-x-8 top-0 h-px"
        aria-hidden="true"
        style={{ backgroundImage: `linear-gradient(90deg, transparent, ${rgba(tone.rgb, 0.45)}, transparent)` }}
      />

      <div className="p-6 sm:p-8">
        {/* Meta row */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <Chip icon={itMeta.Icon} title="Detected input type">{itMeta.label}</Chip>
            {timing && (
              <Chip icon={Timer} title="Total analysis time">
                <span className="tabular-nums">{timing}</span>
                <span className="text-zinc-400">· 6 parallel checks</span>
              </Chip>
            )}
          </div>
          {copyText && <CopyButton text={copyText} />}
        </div>

        <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_auto] lg:gap-x-12 lg:gap-y-0">
          {/* Verdict heading */}
          <div className="min-w-0 lg:col-start-1 lg:row-start-1">
            <p className={`micro-label ${tone.text}`}>Final verdict</p>
            <h3
              ref={headingRef}
              tabIndex={-1}
              className="mt-3 flex items-center gap-3 rounded-lg text-[30px] font-semibold leading-[1.1] tracking-[-0.03em] text-zinc-950 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900/30 focus-visible:ring-offset-4 sm:gap-4 sm:text-5xl"
            >
              <span
                className={`flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-2xl ring-1 sm:h-14 sm:w-14 ${tone.tile}`}
                style={{ boxShadow: `0 8px 24px -10px ${rgba(tone.rgb, 0.55)}` }}
                aria-hidden="true"
              >
                <Icon className="h-5 w-5 sm:h-7 sm:w-7" />
              </span>
              <span className="min-w-0 break-words">
                <span className="sr-only">Final verdict: </span>
                {tone.display}
              </span>
            </h3>
            <p className={`mt-3 text-sm font-medium ${tone.text}`}>{tone.tagline}</p>
          </div>

          {/* Confidence gauge */}
          <div className="lg:col-start-2 lg:row-span-2 lg:row-start-1 lg:self-center">
            <ConfidenceGauge confidence={confidence} accentRgb={tone.rgb} />
          </div>

          {/* Reason + excerpt */}
          <div className="min-w-0 lg:col-start-1 lg:row-start-2 lg:mt-6">
            {explanation && (
              <p className="max-w-prose text-base leading-7 text-zinc-700 sm:text-[17px]">{explanation}</p>
            )}
            {excerpt && <Excerpt text={excerpt} accentRgb={tone.rgb} />}
          </div>
        </div>
      </div>

      {/* Evidence breakdown */}
      <div className="border-t border-zinc-200/70 bg-zinc-50/50 p-6 sm:p-8">
        <EvidenceBreakdown
          verificationWeights={verificationWeights}
          agreementScore={agreementScore}
          robustnessScore={robustnessScore}
          transformerLabel={transformerLabel}
          hybridLabel={hybridLabel}
          accent={accent}
        />
      </div>
    </motion.article>
  );
};

export default VerdictPanel;
