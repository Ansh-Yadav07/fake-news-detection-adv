import React from 'react';
import AnalysisCard from './AnalysisCard';
import { AlertTriangle, AlignLeft, BookOpen, CheckCircle2, Cpu, Info, ListChecks, Newspaper, Type } from 'lucide-react';

// Presentational heuristics only: pick an icon by topic and a tone by wording.
const TOPICS = [
  { re: /wikipedia/i, Icon: BookOpen },
  { re: /\b(news|article|outlet|source|coverage|report)/i, Icon: Newspaper },
  { re: /\b(model|transformer|confidence|lr\b)/i, Icon: Cpu },
  { re: /clickbait|uppercase|capitali[sz]ation|punctuation|sensational/i, Icon: Type },
  { re: /\bwords?\b|text length|short text/i, Icon: AlignLeft },
];

const WARN = /contradict|high clickbait|moderate clickbait|unusually|elevated|above-average|disagree|unavailable|neither|could not|not counted|no relevant|is low|less reliable|limited|likely fake|suspicious|misleading/i;
const GOOD = /confirms|matches|both ai models agree|verified|corroborat|supported|trusted|sufficient|low clickbait|measured/i;

const TONES = {
  warn: 'bg-amber-50 text-amber-700 ring-amber-600/15',
  good: 'bg-emerald-50 text-emerald-700 ring-emerald-600/15',
  neutral: 'bg-zinc-100 text-zinc-600 ring-zinc-500/10',
};

// Strip emoji/pictographs (e.g. a leading warning sign) for a calmer list
const clean = (text) => String(text).replace(/[\p{Extended_Pictographic}️‍]/gu, '').trim();

const classify = (text) => {
  const tone = WARN.test(text) ? 'warn' : GOOD.test(text) ? 'good' : 'neutral';
  const topic = TOPICS.find((t) => t.re.test(text));
  const Icon = topic ? topic.Icon : tone === 'warn' ? AlertTriangle : tone === 'good' ? CheckCircle2 : Info;
  return { tone, Icon };
};

const InsightsList = ({ explanations = [] }) => {
  const items = (Array.isArray(explanations) ? explanations : [])
    .map(clean)
    .filter(Boolean);

  return (
    <AnalysisCard
      title="Analysis insights"
      subtitle="What drove this verdict"
      icon={ListChecks}
      aside={
        items.length > 0 && (
          <span className="inline-flex h-6 items-center rounded-full border border-zinc-200/70 bg-zinc-50 px-2.5 text-xs font-medium text-zinc-600 tabular-nums">
            {items.length}
          </span>
        )
      }
    >
      {items.length > 0 ? (
        <ul className="grid gap-x-10 md:grid-cols-2">
          {items.map((text, idx) => {
            const { tone, Icon } = classify(text);
            return (
              <li key={idx} className="flex items-start gap-3 border-t border-zinc-200/70 py-4 first:border-t-0 md:[&:nth-child(2)]:border-t-0">
                <span className={`mt-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-lg ring-1 ${TONES[tone]}`}>
                  <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                </span>
                <span className="min-w-0 break-words text-sm leading-6 text-zinc-700">{text}</span>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="text-sm text-zinc-500">No additional insights were generated for this text.</p>
      )}
    </AnalysisCard>
  );
};

export default InsightsList;
