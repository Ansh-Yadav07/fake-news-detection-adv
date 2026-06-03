import React from 'react';

const AnalysisCard = ({ title, icon: Icon, children, className = '' }) => {
  return (
    <div className={`bg-white/60 border border-black/5 rounded-xl md:rounded-2xl p-3 md:p-6 shadow-sm flex flex-col h-full hover:shadow-md transition-shadow duration-300 ${className}`}>
      {title && (
        <div className="flex items-center gap-1.5 md:gap-2 mb-2 md:mb-4 pb-2 md:pb-4 border-b border-black/5">
          {Icon && <Icon className="w-3.5 h-3.5 md:w-5 md:h-5 text-zinc-600" />}
          <h4 className="font-semibold text-zinc-900 tracking-tight text-[10px] md:text-sm uppercase">{title}</h4>
        </div>
      )}
      <div className="flex-1">
        {children}
      </div>
    </div>
  );
};

export default AnalysisCard;