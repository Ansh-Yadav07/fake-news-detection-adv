import React, { useId, useState } from 'react';
import AnalysisCard from './AnalysisCard';
import Ring from './Ring';
import {
  Award,
  BookOpen,
  ChevronDown,
  CloudOff,
  ExternalLink,
  Globe,
  Newspaper,
  ShieldAlert,
  ShieldCheck,
  ShieldQuestion,
} from 'lucide-react';

const POSITIVE = { pill: 'bg-emerald-50 text-emerald-700 ring-emerald-600/15', color: '#10b981', icon: ShieldCheck };
const SUPPORTED = { pill: 'bg-sky-50 text-sky-700 ring-sky-600/15', color: '#0ea5e9', icon: ShieldCheck };
const PARTIAL = { pill: 'bg-amber-50 text-amber-700 ring-amber-600/15', color: '#f59e0b', icon: ShieldQuestion };
const NEUTRAL = { pill: 'bg-zinc-100 text-zinc-600 ring-zinc-500/15', color: '#71717a', icon: ShieldQuestion };
const NEGATIVE = { pill: 'bg-red-50 text-red-700 ring-red-600/15', color: '#ef4444', icon: ShieldAlert };

const statusConfig = {
  VERIFIED: POSITIVE,
  'VERIFIED FACT': { ...POSITIVE, icon: BookOpen },
  'LIKELY SUPPORTED': SUPPORTED,
  'PARTIALLY VERIFIED': PARTIAL,
  'PARTIALLY SUPPORTED': PARTIAL,
  'WEAK MATCH': PARTIAL,
  UNVERIFIED: NEUTRAL,
  'NOT FOUND': NEUTRAL,
  'NOT VERIFIED': { ...NEGATIVE, pill: 'bg-orange-50 text-orange-700 ring-orange-600/15', color: '#f97316' },
  SUSPICIOUS: NEGATIVE,
  CONTRADICTED: NEGATIVE,
};

const sourceLabel = (source) => {
  if (source === 'wikipedia') return { text: 'Wikipedia', Icon: BookOpen };
  if (source === 'gnews') return { text: 'News', Icon: Newspaper };
  if (source === 'both') return { text: 'Wikipedia + News', Icon: Globe };
  return { text: 'Online', Icon: Globe };
};

const toTitle = (s) => String(s).toLowerCase().replace(/^\w/, (c) => c.toUpperCase());

const domainOf = (url) => {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return null;
  }
};

/** 16px site favicon via Google's favicon service, with a neutral fallback. */
const SourceFavicon = ({ url, domain: fixedDomain }) => {
  const [failed, setFailed] = useState(false);
  const domain = fixedDomain || domainOf(url);

  return (
    <span className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg border border-zinc-200/70 bg-white">
      {domain && !failed ? (
        <img
          src={`https://www.google.com/s2/favicons?domain=${encodeURIComponent(domain)}&sz=32`}
          alt=""
          width={16}
          height={16}
          loading="lazy"
          decoding="async"
          referrerPolicy="no-referrer"
          className="h-4 w-4 rounded-sm"
          onError={() => setFailed(true)}
        />
      ) : (
        <Globe className="h-4 w-4 text-zinc-400" aria-hidden="true" />
      )}
    </span>
  );
};

const MiniStat = ({ label, value, icon: Icon }) => (
  <div className="bg-white px-4 py-3">
    <dt className="micro-label">{label}</dt>
    <dd className="mt-1.5 flex items-center gap-1.5 text-xl font-semibold leading-none tracking-tight text-zinc-950 tabular-nums">
      {value}
      {Icon}
    </dd>
  </div>
);

const SubHeading = ({ children }) => <h5 className="micro-label mb-3">{children}</h5>;

const UnavailableState = () => (
  <AnalysisCard title="Online verification" subtitle="Wikipedia and live news cross-check" icon={Globe}>
    <div className="flex flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-zinc-200 px-6 py-10 text-center">
      <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-zinc-100 text-zinc-500">
        <CloudOff className="h-5 w-5" aria-hidden="true" />
      </span>
      <p className="mt-4 text-sm font-medium text-zinc-900">Verification unavailable</p>
      <p className="mt-1 max-w-xs text-[13px] leading-5 text-zinc-500">
        No Wikipedia or news evidence was returned for this text, so it was not used in the verdict.
      </p>
    </div>
  </AnalysisCard>
);

const OnlineVerification = ({ wikipedia, verification, verificationSource }) => {
  const [showArticles, setShowArticles] = useState(false);
  const articlesListId = useId();

  // Neither source returned data: explicit unavailable state (not a 0% gauge)
  if (!wikipedia && !verification) return <UnavailableState />;


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

  const config = statusConfig[primaryStatus] || statusConfig.UNVERIFIED;
  const StatusIcon = config.icon;
  const srcInfo = sourceLabel(verificationSource);
  const score = Math.round(Number(primaryScore) || 0);
  const shownArticles = articles.slice(0, 6);
  const wikiScore = Math.round(Number(wikipedia?.verification_score) || 0);
  const wikiStatus = wikipedia?.status ? toTitle(wikipedia.status) : null;

  return (
    <AnalysisCard
      title="Online verification"
      subtitle="Wikipedia and live news cross-check"
      icon={Globe}
      aside={
        <span className="inline-flex h-7 items-center gap-1.5 rounded-full border border-zinc-200/70 bg-zinc-50 px-2.5 text-xs font-medium text-zinc-600">
          <srcInfo.Icon className="h-3.5 w-3.5 text-zinc-400" aria-hidden="true" />
          {srcInfo.text}
        </span>
      }
    >
      <div className="flex flex-1 flex-col gap-6">
        {/* Combined score */}
        <div className="flex items-center gap-4">
          <Ring
            value={score}
            label={`Verification score ${score}%, status ${primaryStatus}`}
            color={config.color}
            strokeWidth={8}
            sizeClass="h-16 w-16 sm:h-[72px] sm:w-[72px]"
          >
            <span className="flex items-start text-lg font-semibold leading-none tracking-tight text-zinc-950 tabular-nums">
              {score}
              <span className="ml-px text-[11px] text-zinc-400">%</span>
            </span>
          </Ring>
          <div className="min-w-0">
            <span className={`inline-flex h-7 items-center gap-1.5 rounded-lg px-2.5 text-[11px] font-semibold uppercase tracking-[0.06em] ring-1 ${config.pill}`}>
              <StatusIcon className="h-3.5 w-3.5" aria-hidden="true" />
              {primaryStatus}
            </span>
            <p className="mt-1.5 text-[13px] text-zinc-500">Combined verification score</p>
          </div>
        </div>

        {/* Wikipedia */}
        {wikipedia && (
          <div>
            <SubHeading>Wikipedia</SubHeading>
            <div className="rounded-xl border border-zinc-200/70 p-4">
              <div className="flex items-center gap-3">
                <SourceFavicon domain="wikipedia.org" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[13px] font-medium text-zinc-900">
                    {wikipedia.wiki_title || 'Wikipedia'}
                  </p>
                  {wikiStatus && <p className="text-xs text-zinc-500">{wikiStatus}</p>}
                </div>
                <span className="text-sm font-semibold text-zinc-900 tabular-nums">
                  {wikiScore}
                  <span className="text-zinc-400">%</span>
                  <span className="sr-only"> Wikipedia score</span>
                </span>
              </div>
              {hasWiki && wikipedia.wiki_extract && (
                <p className="mt-3 line-clamp-3 text-[13px] leading-5 text-zinc-600">{wikipedia.wiki_extract}</p>
              )}
              {wikipedia.wiki_url && (
                <a
                  href={wikipedia.wiki_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="-ml-2 mt-1 inline-flex min-h-10 items-center gap-1.5 rounded-lg px-2 text-[13px] font-medium text-zinc-700 hover:text-zinc-950 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900/40"
                >
                  Read on Wikipedia
                  <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                  <span className="sr-only">(opens in a new tab)</span>
                </a>
              )}
            </div>
          </div>
        )}

        {/* News coverage */}
        {verification && (
          <div className="flex flex-1 flex-col">
            <SubHeading>News coverage</SubHeading>
            <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-zinc-200/70 bg-zinc-200/70">
              <MiniStat label="Supporting" value={supportingCount} />
              <MiniStat
                label="Trusted"
                value={trustedCount}
                icon={
                  trustedCount >= 3 ? (
                    <Award className="h-4 w-4 text-emerald-600" aria-label="Three or more trusted sources" />
                  ) : null
                }
              />
            </dl>

            {articles.length > 0 ? (
              <div className="mt-4">
                <button
                  type="button"
                  onClick={() => setShowArticles((v) => !v)}
                  aria-expanded={showArticles}
                  aria-controls={articlesListId}
                  className="flex min-h-11 w-full items-center justify-between rounded-xl border border-zinc-200/70 bg-white px-4 text-[13px] font-medium text-zinc-700 transition-colors hover:border-zinc-300 hover:bg-zinc-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900/40"
                >
                  <span>
                    {showArticles ? 'Hide sources' : 'Show sources'}
                    <span className="ml-1.5 text-zinc-400 tabular-nums">{articles.length}</span>
                  </span>
                  <ChevronDown
                    className={`h-4 w-4 text-zinc-400 transition-transform duration-200 ${showArticles ? 'rotate-180' : ''}`}
                    aria-hidden="true"
                  />
                </button>

                <div id={articlesListId} hidden={!showArticles}>
                  <ul className="mt-2 divide-y divide-zinc-200/70 animate-in fade-in slide-in-from-top-1 duration-300">
                    {shownArticles.map((article, idx) => {
                      const similarity = Math.round(Number(article.similarity) || 0);
                      const domain = domainOf(article.url);
                      return (
                        <li key={article.url || idx}>
                          <a
                            href={article.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="group -mx-2 flex min-h-11 items-start gap-3 rounded-lg px-2 py-3 transition-colors hover:bg-zinc-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900/40"
                          >
                            <SourceFavicon url={article.url} />
                            <span className="min-w-0 flex-1">
                              <span className="line-clamp-2 text-[13px] font-medium leading-5 text-zinc-800 group-hover:text-zinc-950">
                                {article.title || domain || 'Untitled article'}
                              </span>
                              <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1">
                                <span className="text-xs text-zinc-500">{article.source || domain || 'Unknown source'}</span>
                                {article.is_trusted && (
                                  <span className="inline-flex h-5 items-center gap-1 rounded-md bg-emerald-50 px-1.5 text-[11px] font-semibold text-emerald-700 ring-1 ring-emerald-600/15">
                                    <ShieldCheck className="h-3 w-3" aria-hidden="true" />
                                    Trusted
                                  </span>
                                )}
                                {article.similarity != null && (
                                  <span className="inline-flex items-center gap-1.5 text-xs text-zinc-500 tabular-nums">
                                    <span className="h-1 w-10 overflow-hidden rounded-full bg-zinc-100" aria-hidden="true">
                                      <span className="block h-full rounded-full bg-zinc-400" style={{ width: `${Math.min(100, Math.max(0, similarity))}%` }} />
                                    </span>
                                    {similarity}% match
                                  </span>
                                )}
                              </span>
                            </span>
                            <ExternalLink className="mt-1 h-3.5 w-3.5 flex-shrink-0 text-zinc-300 group-hover:text-zinc-500" aria-hidden="true" />
                            <span className="sr-only">(opens in a new tab)</span>
                          </a>
                        </li>
                      );
                    })}
                  </ul>
                  {articles.length > shownArticles.length && (
                    <p className="mt-2 text-xs text-zinc-500">
                      Showing {shownArticles.length} of {articles.length} sources.
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <p className="mt-4 text-[13px] text-zinc-500">No matching news articles were found.</p>
            )}
          </div>
        )}
      </div>
    </AnalysisCard>
  );
};

export default OnlineVerification;
