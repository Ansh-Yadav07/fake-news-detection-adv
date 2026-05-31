import React from 'react';
import { AlertCircle, CheckCircle2, HelpCircle, ShieldCheck, AlertTriangle, ShieldAlert } from 'lucide-react';

const verdictConfig = {
  'VERIFIED': {
    Icon: ShieldCheck,
    colorClass: 'text-emerald-900 bg-emerald-50 border-emerald-200',
    badgeClass: 'bg-emerald-700 text-white',
    barColor: 'bg-emerald-700',
  },
  'LIKELY REAL': {
    Icon: CheckCircle2,
    colorClass: 'text-sky-900 bg-sky-50 border-sky-200',
    badgeClass: 'bg-sky-700 text-white',
    barColor: 'bg-sky-700',
  },
  'REAL': {
    Icon: CheckCircle2,
    colorClass: 'text-zinc-900 bg-zinc-100 border-zinc-200',
    badgeClass: 'bg-zinc-900 text-white',
    barColor: 'bg-zinc-900',
  },
  'UNVERIFIED': {
    Icon: HelpCircle,
    colorClass: 'text-amber-900 bg-amber-50 border-amber-200',
    badgeClass: 'bg-amber-600 text-white',
    barColor: 'bg-amber-600',
  },
  'UNCERTAIN': {
    Icon: HelpCircle,
    colorClass: 'text-zinc-800 bg-zinc-50 border-zinc-200',
    badgeClass: 'bg-zinc-700 text-white',
    barColor: 'bg-zinc-700',
  },
  'SUSPICIOUS': {
    Icon: AlertTriangle,
    colorClass: 'text-orange-900 bg-orange-50 border-orange-200',
    badgeClass: 'bg-orange-600 text-white',
    barColor: 'bg-orange-600',
  },
  'LIKELY FAKE': {
    Icon: ShieldAlert,
    colorClass: 'text-red-900 bg-red-50 border-red-200',
    badgeClass: 'bg-red-700 text-white',
    barColor: 'bg-red-700',
  },
  'FAKE': {
    Icon: AlertCircle,
    colorClass: 'text-zinc-900 bg-zinc-100 border-zinc-200',
    badgeClass: 'bg-zinc-900 text-white',
    barColor: 'bg-zinc-900',
  },
};

const VerdictPanel = ({ label, transformerLabel, hybridLabel, explanation, agreementScore, robustnessScore, verificationWeights }) => {
  const config = verdictConfig[label] || verdictConfig['UNCERTAIN'];
  const { Icon, colorClass, badgeClass, barColor } = config;

  return (
    <div className={`border rounded-2xl p-6 md:p-8 flex flex-col md:flex-row gap-8 items-center shadow-sm relative overflow-hidden ${colorClass}`}>
      {/* Decorative background circle */}
      <div className="absolute -right-6 -top-6 w-32 h-32 rounded-full opacity-5 bg-current pointer-events-none" />
      
      <div className="flex-shrink-0 flex flex-col items-center justify-center text-center">
        <div className="text-xs font-bold tracking-widest uppercase mb-3 opacity-70">Final Verdict</div>
        <div className={`px-6 py-3 rounded-2xl text-2xl md:text-3xl font-black tracking-tight shadow-sm flex items-center gap-3 ${badgeClass}`}>
          <Icon className="w-8 h-8" />
          {label}
        </div>
      </div>

      <div className="flex-1 space-y-5">
        <p className="text-sm md:text-base font-medium opacity-90 leading-relaxed">
          {explanation}
        </p>

        <div className="grid grid-cols-2 gap-6 pt-4 border-t border-current/10">
          <div>
            <div className="flex justify-between text-xs font-bold uppercase tracking-wider mb-2 opacity-70">
              <span>Model Agreement</span>
              <span>{agreementScore}%</span>
            </div>
            <div className="h-1.5 w-full bg-black/10 rounded-full overflow-hidden">
              <div className={`h-full ${barColor} rounded-full transition-all duration-1000`} style={{ width: `${agreementScore}%` }} />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-xs font-bold uppercase tracking-wider mb-2 opacity-70">
              <span>Robustness Score</span>
              <span>{robustnessScore}/10</span>
            </div>
            <div className="h-1.5 w-full bg-black/10 rounded-full overflow-hidden">
              <div className={`h-full ${barColor} rounded-full transition-all duration-1000`} style={{ width: `${robustnessScore * 10}%` }} />
            </div>
          </div>
        </div>

        {/* Verification Weight Breakdown */}
        {verificationWeights && (
          <div className="pt-4 border-t border-current/10">
            <div className="text-xs font-bold uppercase tracking-wider mb-3 opacity-70">
              Signal Weights
            </div>
            <div className="flex gap-1.5 h-2 rounded-full overflow-hidden bg-black/5">
              <div
                className="rounded-l-full bg-emerald-500 transition-all duration-1000"
                style={{ width: `${verificationWeights.verification * 0.5}%` }}
                title={`Verification: ${verificationWeights.verification}%`}
              />
              <div
                className="bg-sky-500 transition-all duration-1000"
                style={{ width: `${verificationWeights.ml * 0.2}%` }}
                title={`ML Models: ${verificationWeights.ml}%`}
              />
              <div
                className="bg-violet-500 transition-all duration-1000"
                style={{ width: `${verificationWeights.credibility * 0.15}%` }}
                title={`Credibility: ${verificationWeights.credibility}%`}
              />
              <div
                className="bg-amber-500 transition-all duration-1000"
                style={{ width: `${verificationWeights.linguistic * 0.1}%` }}
                title={`Linguistic: ${verificationWeights.linguistic}%`}
              />
              <div
                className="rounded-r-full bg-zinc-400 transition-all duration-1000"
                style={{ width: `${verificationWeights.clickbait * 0.05}%` }}
                title={`Clickbait: ${verificationWeights.clickbait}%`}
              />
            </div>
            <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
              <span className="text-[10px] font-semibold text-zinc-500 flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" /> Verification {verificationWeights.verification}%
              </span>
              <span className="text-[10px] font-semibold text-zinc-500 flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-sky-500 inline-block" /> ML {verificationWeights.ml}%
              </span>
              <span className="text-[10px] font-semibold text-zinc-500 flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-violet-500 inline-block" /> Credibility {verificationWeights.credibility}%
              </span>
              <span className="text-[10px] font-semibold text-zinc-500 flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-amber-500 inline-block" /> Linguistic {verificationWeights.linguistic}%
              </span>
              <span className="text-[10px] font-semibold text-zinc-500 flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-zinc-400 inline-block" /> Clickbait {verificationWeights.clickbait}%
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default VerdictPanel;