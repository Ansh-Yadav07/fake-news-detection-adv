import React, { useState } from 'react';
import AnalysisCard from './AnalysisCard';
import { Globe, ShieldCheck, ShieldAlert, ShieldQuestion, ExternalLink, ChevronDown, ChevronUp, Award } from 'lucide-react';

const statusConfig = {
  'VERIFIED': {
    color: 'text-emerald-700',
    bg: 'bg-emerald-50',
    border: 'border-emerald-200',
    badgeBg: 'bg-emerald-600',
    ringColor: 'ring-emerald-400',
    icon: ShieldCheck,
    glowColor: 'rgba(16, 185, 129, 0.15)',
  },
  'LIKELY SUPPORTED': {
    color: 'text-sky-700',
    bg: 'bg-sky-50',
    border: 'border-sky-200',
    badgeBg: 'bg-sky-600',
    ringColor: 'ring-sky-400',
    icon: ShieldCheck,
    glowColor: 'rgba(14, 165, 233, 0.15)',
  },
  'PARTIALLY SUPPORTED': {
    color: 'text-amber-700',
    bg: 'bg-amber-50',
    border: 'border-amber-200',
    badgeBg: 'bg-amber-600',
    ringColor: 'ring-amber-400',
    icon: ShieldQuestion,
    glowColor: 'rgba(245, 158, 11, 0.15)',
  },
  'UNVERIFIED': {
    color: 'text-zinc-600',
    bg: 'bg-zinc-50',
    border: 'border-zinc-200',
    badgeBg: 'bg-zinc-500',
    ringColor: 'ring-zinc-400',
    icon: ShieldQuestion,
    glowColor: 'rgba(113, 113, 122, 0.15)',
  },
  'SUSPICIOUS': {
    color: 'text-red-700',
    bg: 'bg-red-50',
    border: 'border-red-200',
    badgeBg: 'bg-red-600',
    ringColor: 'ring-red-400',
    icon: ShieldAlert,
    glowColor: 'rgba(239, 68, 68, 0.15)',
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

const OnlineVerification = ({ verification }) => {
  const [showArticles, setShowArticles] = useState(false);

  if (!verification) return null;

  const {
    verification_score = 0,
    supporting_articles = 0,
    trusted_source_count = 0,
    status = 'UNVERIFIED',
    articles = [],
  } = verification;

  const config = statusConfig[status] || statusConfig['UNVERIFIED'];
  const StatusIcon = config.icon;

  const progressColor = status === 'VERIFIED' ? '#10b981'
    : status === 'LIKELY SUPPORTED' ? '#0ea5e9'
    : status === 'SUSPICIOUS' ? '#ef4444'
    : status === 'PARTIALLY SUPPORTED' ? '#f59e0b'
    : '#71717a';

  return (
    <AnalysisCard title="Online Verification" icon={Globe}>
      <div className="flex flex-col h-full">
        {/* Score Circle + Status */}
        <div className="flex items-center gap-5 mb-5">
          <div className="relative flex-shrink-0">
            <CircularProgress value={verification_score} color={progressColor} />
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-lg font-black text-zinc-900">{Math.round(verification_score)}%</span>
            </div>
          </div>

          <div className="flex-1 min-w-0">
            <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold tracking-wide uppercase ${config.bg} ${config.color} ${config.border} border mb-2`}>
              <StatusIcon className="w-3.5 h-3.5" />
              {status}
            </div>
            <p className="text-xs text-zinc-500 font-medium leading-snug">
              Verification Score
            </p>
          </div>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="bg-zinc-50 border border-zinc-100 rounded-xl p-3 text-center">
            <div className="text-xl font-bold text-zinc-900">{supporting_articles}</div>
            <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mt-0.5">Supporting</div>
          </div>
          <div className="bg-zinc-50 border border-zinc-100 rounded-xl p-3 text-center">
            <div className="text-xl font-bold text-zinc-900 flex items-center justify-center gap-1">
              {trusted_source_count}
              {trusted_source_count >= 3 && <Award className="w-4 h-4 text-emerald-500" />}
            </div>
            <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mt-0.5">Trusted</div>
          </div>
        </div>

        {/* Expandable Articles List */}
        {articles.length > 0 && (
          <div className="mt-auto">
            <button
              onClick={() => setShowArticles(!showArticles)}
              className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-zinc-50 border border-zinc-100 hover:border-zinc-200 transition-colors text-xs font-semibold text-zinc-600"
            >
              <span>Source Matches ({articles.length})</span>
              {showArticles ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
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
