import React from 'react';
import AnalysisCard from './AnalysisCard';
import { Type } from 'lucide-react';

const StatRing = ({ label, value, color = '#18181b', trackColor = '#f4f4f5', size, strokeWidth }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  // Decorative fill (75%) as a visual accent
  const offset = circumference * 0.25;
  const center = size / 2;

  return (
    <div className="flex flex-col items-center gap-1 md:gap-1.5">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke={trackColor}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
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
            opacity="0.2"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xs md:text-lg font-black text-zinc-900 leading-none">{value}</span>
        </div>
      </div>
      <span className="text-[8px] md:text-[10px] font-semibold text-zinc-500 uppercase tracking-wider text-center leading-tight">{label}</span>
    </div>
  );
};

const StatCell = ({ label, value, color, mobileSize, desktopSize }) => (
  <>
    <div className="md:hidden">
      <StatRing label={label} value={value} color={color} size={mobileSize} strokeWidth={4} />
    </div>
    <div className="hidden md:block">
      <StatRing label={label} value={value} color={color} size={desktopSize} strokeWidth={5} />
    </div>
  </>
);

const StatsGrid = ({ wordCount, avgWordLength, upperRatio, punctDensity }) => {
  return (
    <AnalysisCard title="Text Statistics" icon={Type}>
      <div className="grid grid-cols-2 gap-2 md:gap-4 h-full place-items-center py-1 md:py-2">
        <StatCell label="Words" value={wordCount} color="#18181b" mobileSize={52} desktopSize={72} />
        <StatCell label="Avg Length" value={avgWordLength} color="#52525b" mobileSize={52} desktopSize={72} />
        <StatCell label="Uppercase" value={`${upperRatio}%`} color="#71717a" mobileSize={52} desktopSize={72} />
        <StatCell label="Punctuation" value={`${punctDensity}%`} color="#a1a1aa" mobileSize={52} desktopSize={72} />
      </div>
    </AnalysisCard>
  );
};

export default StatsGrid;