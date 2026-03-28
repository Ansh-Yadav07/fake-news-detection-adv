import React from 'react';
import { AlertCircle, CheckCircle2, HelpCircle } from 'lucide-react';

const VerdictPanel = ({ label, transformerLabel, hybridLabel, explanation, agreementScore, robustnessScore }) => {
  const isReal = label === 'REAL';
  const isUncertain = label === 'UNCERTAIN' || transformerLabel !== hybridLabel;
  const isFake = !isReal && !isUncertain;

  let Icon = AlertCircle;
  let colorClass = 'text-zinc-900 bg-zinc-100 border-zinc-200';
  let badgeClass = 'bg-zinc-900 text-white';
  
  if (isReal) {
    Icon = CheckCircle2;
  } else if (isUncertain) {
    Icon = HelpCircle;
    colorClass = 'text-zinc-800 bg-zinc-50 border-zinc-200';
    badgeClass = 'bg-zinc-700 text-white';
  }

  const finalVerdict = isUncertain ? 'UNCERTAIN' : label;

  return (
    <div className={`border rounded-2xl p-6 md:p-8 flex flex-col md:flex-row gap-8 items-center shadow-sm relative overflow-hidden ${colorClass}`}>
      {/* Decorative background circle */}
      <div className="absolute -right-6 -top-6 w-32 h-32 rounded-full opacity-5 bg-current pointer-events-none" />
      
      <div className="flex-shrink-0 flex flex-col items-center justify-center text-center">
        <div className="text-xs font-bold tracking-widest uppercase mb-3 opacity-70">Final Verdict</div>
        <div className={`px-6 py-3 rounded-2xl text-2xl md:text-3xl font-black tracking-tight shadow-sm flex items-center gap-3 ${badgeClass}`}>
          <Icon className="w-8 h-8" />
          {finalVerdict}
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
              <div className="h-full bg-zinc-900 rounded-full" style={{ width: `${agreementScore}%` }} />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-xs font-bold uppercase tracking-wider mb-2 opacity-70">
              <span>Robustness Score</span>
              <span>{robustnessScore}/10</span>
            </div>
            <div className="h-1.5 w-full bg-black/10 rounded-full overflow-hidden">
              <div className="h-full bg-zinc-900 rounded-full" style={{ width: `${robustnessScore * 10}%` }} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VerdictPanel;