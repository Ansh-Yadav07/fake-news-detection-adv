import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import AnalysisCard from './AnalysisCard';
import { BarChart2 } from 'lucide-react';

const ChartSection = ({ transformerConf, hybridConf }) => {
  const data = [
    { name: 'Transformer', confidence: transformerConf },
    { name: 'Hybrid ML', confidence: hybridConf }
  ];

  return (
    <AnalysisCard title="Confidence Models" icon={BarChart2}>
      <div className="h-40 w-full mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top:0, right: 0, left:20, bottom: 0 }} layout="vertical">
            <XAxis type="number" domain={[0, 100]} hide />
            <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fill: '#52525b', fontSize: 12, fontWeight: 500 }} />
            <Tooltip 
              cursor={{fill: 'transparent'}}
              contentStyle={{ borderRadius: '12px', border: '1px solid rgba(0,0,0,0.1)', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
            />
            <Bar dataKey="confidence" radius={[0, 8, 8, 0]} barSize={24}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.name === 'Transformer' ? '#18181b' : '#71717a'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="flex justify-between mt-4 text-xs font-medium text-zinc-500 px-2">
        <span></span>
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
    </AnalysisCard>
  );
};

export default ChartSection;