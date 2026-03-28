import React from 'react';
import AnalysisCard from './AnalysisCard';
import { ListChecks } from 'lucide-react';

const InsightsList = ({ explanations }) => {
  return (
    <AnalysisCard title="AI Analysis Insights" icon={ListChecks} className="md:col-span-2">
      <ul className="grid md:grid-cols-2 gap-4">
        {explanations.map((exp, idx) => (
          <li key={idx} className="flex items-start gap-3 p-3 rounded-xl bg-zinc-50 border border-zinc-100 hover:border-zinc-200 transition-colors">
            <div className="mt-1.5 w-1.5 h-1.5 rounded-full bg-zinc-400 flex-shrink-0" />
            <span className="text-sm font-medium text-zinc-700 leading-relaxed">{exp}</span>
          </li>
        ))}
      </ul>
    </AnalysisCard>
  );
};

export default InsightsList;