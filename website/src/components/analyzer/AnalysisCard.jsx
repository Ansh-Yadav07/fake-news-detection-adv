import React from 'react';

const AnalysisCard = ({ title, icon: Icon, children, className = '' }) => {
  return (
    <div className={`bg-white/60 border border-black/5 rounded-2xl p-6 shadow-sm flex flex-col h-full hover:shadow-md transition-shadow duration-300 ${className}`}>
      {title && (
        <div className="flex items-center gap-2 mb-4 pb-4 border-b border-black/5">
          {Icon && <Icon className="w-5 h-5 text-zinc-600" />}
          <h4 className="font-semibold text-zinc-900 tracking-tight text-sm uppercase">{title}</h4>
        </div>
      )}
      <div className="flex-1">
        {children}
      </div>
    </div>
  );
};

export default AnalysisCard;