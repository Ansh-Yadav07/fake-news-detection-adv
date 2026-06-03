import React, { useState } from 'react';
import AnalysisCard from './AnalysisCard';
import { Globe, ShieldCheck, ShieldAlert, ShieldQuestion, ExternalLink, ChevronDown, ChevronUp, Award, BookOpen, Newspaper } from 'lucide-react';

const statusConfig = {
  'VERIFIED': {
    color: 'text-emerald-700',
    bg: 'bg-emerald-50',
    border: 'border-emerald-200',
    badgeBg: 'bg-emerald-600',
    icon: ShieldCheck,
    glowColor: 'rgba(16, 185, 129, 0.15)',
  },
  'VERIFIED FACT': {
    color: 'text-emerald-700',
    bg: 'bg-emerald-50',
    border: 'border-emerald-200',
    badgeBg: 'bg-emerald-700',
    icon: BookOpen,
    glowColor: 'rgba(16, 185, 129, 0.2)',
  },
  'LIKELY SUPPORTED': {
    color: 'text-sky-700',
    bg: 'bg-sky-50',
    border: 'border-sky-200',
    badgeBg: 'bg-sky-600',
    icon: ShieldCheck,
    glowColor: 'rgba(14, 165, 233, 0.15)',
  },
  'PARTIALLY VERIFIED': {
    color: 'text-amber-700',
    bg: 'bg-amber-50',
    border: 'border-amber-200',
    badgeBg: 'bg-amber-600',
    icon: ShieldQuestion,
    glowColor: 'rgba(245, 158, 11, 0.15)',
  },
  'PARTIALLY SUPPORTED': {
    color: 'text-amber-700',
    bg: 'bg-amber-50',
    border: 'border-amber-200',
    badgeBg: 'bg-amber-600',
    icon: ShieldQuestion,
    glowColor: 'rgba(245, 158, 11, 0.15)',
  },
  'UNVERIFIED': {
    color: 'text-zinc-600',
    bg: 'bg-zinc-50',
    border: 'border-zinc-200',
    badgeBg: 'bg-zinc-500',
    icon: ShieldQuestion,
    glowColor: 'rgba(113, 113, 122, 0.15)',
  },
  'NOT FOUND': {
    color: 'text-zinc-500',
    bg: 'bg-zinc-50',
    border: 'border-zinc-200',
    badgeBg: 'bg-zinc-400',
    icon: ShieldQuestion,
    glowColor: 'rgba(113, 113, 122, 0.1)',
  },
  'NOT VERIFIED': {
    color: 'text-orange-700',
    bg: 'bg-orange-50',
    border: 'border-orange-200',
    badgeBg: 'bg-orange-600',
    icon: ShieldAlert,
    glowColor: 'rgba(249, 115, 22, 0.15)',
  },
  'WEAK MATCH': {
    color: 'text-amber-600',
    bg: 'bg-amber-50',
    border: 'border-amber-200',
    badgeBg: 'bg-amber-500',
    icon: ShieldQuestion,
    glowColor: 'rgba(245, 158, 11, 0.15)',
  },
  'SUSPICIOUS': {
    color: 'text-red-700',
    bg: 'bg-red-50',
    border: 'border-red-200',
    badgeBg: 'bg-red-600',
    icon: ShieldAlert,
    glowColor: 'rgba(239, 68, 68, 0.15)',
  },
  'CONTRADICTED': {
    color: 'text-red-800',
    bg: 'bg-red-50',
    border: 'border-red-300',
    badgeBg: 'bg-red-700',
    icon: ShieldAlert,
    glowColor: 'rgba(239, 68, 68, 0.2)',
  },
};

const CircularProgress = ({ value, size = 80, strokeWidth = 6, color }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (value / 100) * circumference;

  return (
    <svg width={size} height={size} className="transform -rotate-90">
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        className="text-zinc-100"
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
        className="transition-all duration-1000 ease-out"
      />
    </svg>
  );
};

const ResponsiveCircularProgress = ({ value, color }) => (
  <>
    <div className="md:hidden">
      <CircularProgress value={value} size={56} strokeWidth={5} color={color} />
    </div>
    <div className="hidden md:block">
      <CircularProgress value={value} size={80} strokeWidth={6} color={color} />
    </div>
  </>
);

const sourceLabel = (source) => {
  if (source === 'wikipedia') return { text: 'Wikipedia', Icon: BookOpen, cls: 'text-violet-700 bg-violet-50 border-violet-200' };
  if (source === 'gnews') return { text: 'GNews', Icon: Newspaper, cls: 'text-sky-700 bg-sky-50 border-sky-200' };
  if (source === 'both') return { text: 'Wikipedia + GNews', Icon: Globe, cls: 'text-emerald-700 bg-emerald-50 border-emerald-200' };
  return { text: 'Online', Icon: Globe, cls: 'text-zinc-600 bg-zinc-50 border-zinc-200' };
};

const OnlineVerification = ({ wikipedia, verification, verificationSource, inputType }) => {
  const [showArticles, setShowArticles] = useState(false);

  // If neither source has data, show empty state
  if (!wikipedia && !verification) return null;

  // Determine which data to show based on source
  const hasWiki = wikipedia && wikipedia.status && wikipedia.status !== 'NOT FOUND';
  const hasGNews = verification && (verification.supporting_articles > 0 || (verification.articles && verification.articles.length > 0));

  // Calculate combined verification score
  let primaryScore = 0;
  let primaryStatus = 'UNVERIFIED';
  let articles = [];
  let supportingCount = 0;
  let trustedCount = 0;

  if (hasWiki && !hasGNews) {
    primaryScore = wikipedia.verification_score || 0;
    primaryStatus = wikipedia.status || 'UNVERIFIED';
  } else if (hasGNews && !hasWiki) {
    primaryScore = verification.verification_score || 0;
    primaryStatus = verification.status || 'UNVERIFIED';
    articles = verification.articles || [];
    supportingCount = verification.supporting_articles || 0;
    trustedCount = verification.trusted_source_count || 0;
  } else if (hasWiki && hasGNews) {
    // Both available — show combined score
    const wScore = wikipedia.verification_score || 0;
    const gScore = verification.verification_score || 0;
    primaryScore = Math.round((wScore + gScore) / 2);
    primaryStatus = primaryScore > 70 ? 'VERIFIED' : primaryScore > 40 ? 'PARTIALLY VERIFIED' : 'UNVERIFIED';
    articles = verification.articles || [];
    supportingCount = verification.supporting_articles || 0;
    trustedCount = verification.trusted_source_count || 0;
  } else {
    // Neither has good data — still show what we have from news APIs
    if (wikipedia) {
      primaryScore = wikipedia.verification_score || 0;
      primaryStatus = wikipedia.status || 'NOT FOUND';
    }
    if (verification) {
      // If news returned a score, use the higher of wiki/news
      const newsScore = verification.verification_score || 0;
      if (newsScore > primaryScore) {
        primaryScore = newsScore;
      }
      primaryStatus = verification.status || primaryStatus;
      articles = verification.articles || [];
      supportingCount = verification.supporting_articles || 0;
      trustedCount = verification.trusted_source_count || 0;
    }
  }

  const config = statusConfig[primaryStatus] || statusConfig['UNVERIFIED'];
  const StatusIcon = config.icon;
  const srcInfo = sourceLabel(verificationSource);

  const progressColor = primaryStatus === 'VERIFIED' || primaryStatus === 'VERIFIED FACT' ? '#10b981'
    : primaryStatus === 'LIKELY SUPPORTED' ? '#0ea5e9'
    : primaryStatus === 'SUSPICIOUS' || primaryStatus === 'NOT VERIFIED' || primaryStatus === 'CONTRADICTED' ? '#ef4444'
    : primaryStatus === 'PARTIALLY SUPPORTED' || primaryStatus === 'PARTIALLY VERIFIED' || primaryStatus === 'WEAK MATCH' ? '#f59e0b'
    : '#71717a';

  return (
    <AnalysisCard title="Online Verification" icon={Globe}>
      <div className="flex flex-col h-full">
        {/* Source Badge */}
        <div className={`inline-flex items-center gap-1 md:gap-1.5 px-1.5 md:px-2 py-0.5 md:py-1 rounded-md md:rounded-lg border text-[8px] md:text-[10px] font-bold uppercase tracking-wide mb-2 md:mb-3 self-start ${srcInfo.cls}`}>
          <srcInfo.Icon className="w-2.5 h-2.5 md:w-3 md:h-3" />
          {srcInfo.text}
        </div>

        {/* Score Circle + Status */}
        <div className="flex items-center gap-2 md:gap-5 mb-3 md:mb-5">
          <div className="relative flex-shrink-0">
            <ResponsiveCircularProgress value={primaryScore} color={progressColor} />
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-xs md:text-lg font-black text-zinc-900">{Math.round(primaryScore)}%</span>
            </div>
          </div>

          <div className="flex-1 min-w-0">
            <div className={`inline-flex items-center gap-1 md:gap-1.5 px-1.5 md:px-2.5 py-0.5 md:py-1 rounded-md md:rounded-lg text-[9px] md:text-xs font-bold tracking-wide uppercase ${config.bg} ${config.color} ${config.border} border mb-1 md:mb-2`}>
              <StatusIcon className="w-2.5 h-2.5 md:w-3.5 md:h-3.5" />
              {primaryStatus}
            </div>
            <p className="text-[9px] md:text-xs text-zinc-500 font-medium leading-snug">
              Verification Score
            </p>
          </div>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-2 gap-1.5 md:gap-3 mb-2 md:mb-4">
          {/* Wikipedia stats */}
          {hasWiki && (
            <div className="bg-violet-50 border border-violet-100 rounded-lg md:rounded-xl p-1.5 md:p-3 text-center">
              <div className="text-sm md:text-xl font-bold text-violet-900">{wikipedia.verification_score || 0}%</div>
              <div className="text-[8px] md:text-[10px] font-semibold text-violet-500 uppercase tracking-wider mt-0.5">Wikipedia</div>
            </div>
          )}
          {/* GNews supporting count */}
          {hasGNews && (
            <div className="bg-sky-50 border border-sky-100 rounded-lg md:rounded-xl p-1.5 md:p-3 text-center">
              <div className="text-sm md:text-xl font-bold text-sky-900">{supportingCount}</div>
              <div className="text-[8px] md:text-[10px] font-semibold text-sky-500 uppercase tracking-wider mt-0.5">Supporting</div>
            </div>
          )}
          {/* Trusted sources */}
          {hasGNews && (
            <div className="bg-zinc-50 border border-zinc-100 rounded-lg md:rounded-xl p-1.5 md:p-3 text-center">
              <div className="text-sm md:text-xl font-bold text-zinc-900 flex items-center justify-center gap-1">
                {trustedCount}
                {trustedCount >= 3 && <Award className="w-3 h-3 md:w-4 md:h-4 text-emerald-500" />}
              </div>
              <div className="text-[8px] md:text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mt-0.5">Trusted</div>
            </div>
          )}
          {/* If only wiki but no gnews, fill with wiki info */}
          {hasWiki && !hasGNews && (
            <div className="bg-zinc-50 border border-zinc-100 rounded-lg md:rounded-xl p-1.5 md:p-3 text-center">
              <div className="text-sm md:text-xl font-bold text-zinc-900">1</div>
              <div className="text-[8px] md:text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mt-0.5">Sources</div>
            </div>
          )}
        </div>

        {/* Wikipedia Extract */}
        {hasWiki && wikipedia.wiki_extract && (
          <div className="mb-2 md:mb-3 p-1.5 md:p-2.5 rounded-lg bg-violet-50/50 border border-violet-100 hidden md:block">
            <p className="text-[11px] text-violet-800 leading-relaxed line-clamp-3">
              <span className="font-bold">Wikipedia:</span> {wikipedia.wiki_extract.substring(0, 150)}...
            </p>
            {wikipedia.wiki_url && (
              <a
                href={wikipedia.wiki_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 mt-1.5 text-[10px] font-semibold text-violet-600 hover:text-violet-800 transition-colors"
              >
                Read full article <ExternalLink className="w-2.5 h-2.5" />
              </a>
            )}
          </div>
        )}

        {/* Expandable GNews Articles List */}
        {articles.length > 0 && (
          <div className="mt-auto">
            <button
              onClick={() => setShowArticles(!showArticles)}
              className="w-full flex items-center justify-between px-2 md:px-3 py-1.5 md:py-2 rounded-lg bg-zinc-50 border border-zinc-100 hover:border-zinc-200 transition-colors text-[10px] md:text-xs font-semibold text-zinc-600"
            >
              <span>Sources ({articles.length})</span>
              {showArticles ? <ChevronUp className="w-3 h-3 md:w-3.5 md:h-3.5" /> : <ChevronDown className="w-3 h-3 md:w-3.5 md:h-3.5" />}
            </button>

            {showArticles && (
              <div className="mt-2 space-y-2 max-h-48 overflow-y-auto pr-1 animate-in fade-in slide-in-from-top-2 duration-300">
                {articles.slice(0, 6).map((article, idx) => (
                  <a
                    key={idx}
                    href={article.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-start gap-2 p-2.5 rounded-lg bg-white border border-zinc-100 hover:border-zinc-200 hover:shadow-sm transition-all group"
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-zinc-800 truncate group-hover:text-zinc-900 transition-colors">
                        {article.title}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <span className={`text-[10px] font-bold uppercase tracking-wide ${article.is_trusted ? 'text-emerald-600' : 'text-zinc-400'}`}>
                          {article.source}
                        </span>
                        <span className="text-[10px] font-semibold text-zinc-400">
                          {article.similarity}%
                        </span>
                      </div>
                    </div>
                    <ExternalLink className="w-3 h-3 text-zinc-300 group-hover:text-zinc-500 flex-shrink-0 mt-0.5 transition-colors" />
                  </a>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </AnalysisCard>
  );
};

export default OnlineVerification;
