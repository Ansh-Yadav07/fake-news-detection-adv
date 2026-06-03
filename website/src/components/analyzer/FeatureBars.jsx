import React from 'react';
import AnalysisCard from './AnalysisCard';
import { AlignLeft } from 'lucide-react';

const MiniRadial = ({ label, value, max = 100, isWarning = false, size, strokeWidth }) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percentage / 100) * circumference;
  const center = size / 2;

  // Dynamic color based on severity
  const getColor = () => {
    if (!isWarning) return '#a1a1aa'; // neutral zinc-400
    if (percentage > 70) return '#18181b'; // high — zinc-900
    if (percentage > 40) return '#52525b'; // medium — zinc-600
    return '#a1a1aa'; // low — zinc-400
  };

  const color = getColor();
  const displayValue = max === 100 ? `${value}%` : value;

  return (
    <div className="flex flex-col items-center gap-1 md:gap-1.5 p-1 md:p-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke="#f4f4f5"
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
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-[10px] md:text-sm font-bold text-zinc-900">{displayValue}</span>
        </div>
      </div>
      <span className="text-[8px] md:text-[10px] font-semibold text-zinc-500 text-center leading-tight uppercase tracking-wide">{label}</span>
    </div>
  );
};

const FeatureItem = ({ label, value, max, isWarning, mobileSize, desktopSize }) => (
  <>
    {/* Mobile */}
    <div className="md:hidden">
      <MiniRadial label={label} value={value} max={max} isWarning={isWarning} size={mobileSize} strokeWidth={4} />
    </div>
    {/* Desktop */}
    <div className="hidden md:block">
      <MiniRadial label={label} value={value} max={max} isWarning={isWarning} size={desktopSize} strokeWidth={6} />
    </div>
  </>
);

const FeatureBars = ({ punctuation, uppercase, complexity, clickbait }) => {
  return (
    <AnalysisCard title="Linguistic Features" icon={AlignLeft}>
      <div className="grid grid-cols-2 gap-1 md:gap-2 h-full place-items-center py-0 md:py-1">
        <FeatureItem label="Punctuation" value={punctuation} max={100} isWarning mobileSize={52} desktopSize={72} />
        <FeatureItem label="Uppercase" value={uppercase} max={100} isWarning mobileSize={52} desktopSize={72} />
        <FeatureItem label="Complexity" value={complexity} max={10} isWarning={false} mobileSize={52} desktopSize={72} />
        <FeatureItem label="Clickbait" value={clickbait} max={100} isWarning mobileSize={52} desktopSize={72} />
      </div>
    </AnalysisCard>
  );
};

export default FeatureBars;