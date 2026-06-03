import React from 'react';
import AnalysisCard from './AnalysisCard';
import { BarChart2 } from 'lucide-react';

const CircularGauge = ({ label, value, color, trackColor, size, strokeWidth }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const percentage = Math.min(100, Math.max(0, value));
  const offset = circumference - (percentage / 100) * circumference;
  const center = size / 2;

  return (
    <div className="flex flex-col items-center gap-1 md:gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          {/* Track */}
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke={trackColor}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          {/* Progress */}
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="gauge-progress"
          />
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-base md:text-2xl font-black text-zinc-900 leading-none">
            {Math.round(value)}
          </span>
          <span className="text-[8px] md:text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">%</span>
        </div>
      </div>
      <span className="text-[9px] md:text-xs font-semibold text-zinc-600 text-center leading-tight">{label}</span>
    </div>
  );
};

const ChartSection = ({ transformerConf, hybridConf }) => {
  // Use window check for SSR safety, but we'll use CSS-driven sizing via a wrapper
  return (
    <AnalysisCard title="Confidence Models" icon={BarChart2}>
      {/* Mobile layout */}
      <div className="flex md:hidden items-center justify-around h-full py-1">
        <CircularGauge
          label="Transformer"
          value={transformerConf}
          color="#18181b"
          trackColor="#f4f4f5"
          size={72}
          strokeWidth={7}
        />
        <CircularGauge
          label="Hybrid ML"
          value={hybridConf}
          color="#71717a"
          trackColor="#f4f4f5"
          size={72}
          strokeWidth={7}
        />
      </div>
      {/* Desktop layout */}
      <div className="hidden md:flex items-center justify-around h-full py-3">
        <CircularGauge
          label="Transformer"
          value={transformerConf}
          color="#18181b"
          trackColor="#f4f4f5"
          size={110}
          strokeWidth={10}
        />
        <CircularGauge
          label="Hybrid ML"
          value={hybridConf}
          color="#71717a"
          trackColor="#f4f4f5"
          size={110}
          strokeWidth={10}
        />
      </div>
    </AnalysisCard>
  );
};

export default ChartSection;