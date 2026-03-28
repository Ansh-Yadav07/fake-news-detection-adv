import React from 'react';
import AnalysisCard from './AnalysisCard';
import { Type } from 'lucide-react';

const StatItem = ({ label, value }) => (
  <div className="bg-zinc-50 border border-zinc-100 rounded-xl p-4 flex flex-col justify-center items-center text-center">
    <div className="text-2xl font-bold text-zinc-900 mb-1">{value}</div>
    <div className="text-xs font-medium text-zinc-500 uppercase tracking-wider">{label}</div>
  </div>
);

const StatsGrid = ({ wordCount, avgWordLength, upperRatio, punctDensity }) => {
  return (
    <AnalysisCard title="Text Statistics" icon={Type}>
      <div className="grid grid-cols-2 gap-4 h-full">
        <StatItem label="Words" value={wordCount} />
        <StatItem label="Avg Length" value={avgWordLength} />
        <StatItem label="Uppercase" value={`${upperRatio}%`} />
        <StatItem label="Punctuation" value={`${punctDensity}%`} />
      </div>
    </AnalysisCard>
  );
};

export default StatsGrid;