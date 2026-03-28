import React from 'react';
import AnalysisCard from './AnalysisCard';
import { AlignLeft } from 'lucide-react';

const FeatureBar = ({ label, value, max = 100, isWarning = false }) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  
  return (
    <div className="mb-4 last:mb-0">
      <div className="flex justify-between items-center mb-1.5">
        <span className="text-sm font-medium text-zinc-700">{label}</span>
        <span className="text-xs font-bold text-zinc-900">{value}{(max === 100) ? '%' : ''}</span>
      </div>
      <div className="h-2 w-full bg-zinc-100 rounded-full overflow-hidden">
        <div 
          className={`h-full rounded-full transition-all duration-1000 ${isWarning && value > (max * 0.7) ? 'bg-zinc-800' : 'bg-zinc-400'}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

const FeatureBars = ({ punctuation, uppercase, complexity, clickbait }) => {
  return (
    <AnalysisCard title="Linguistic Features" icon={AlignLeft}>
      <div className="flex flex-col justify-center h-full space-y-4 py-2">
        <FeatureBar label="Punctuation Intensity" value={punctuation} isWarning />
        <FeatureBar label="Uppercase Ratio" value={uppercase} isWarning />
        <FeatureBar label="Text Complexity" value={complexity} max={10} />
        <FeatureBar label="Clickbait Score" value={clickbait} max={100} isWarning />
      </div>
    </AnalysisCard>
  );
};

export default FeatureBars;