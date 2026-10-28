import React, { useId } from 'react';
import { motion } from 'framer-motion';
import { fadeUp } from './motion';

/**
 * Bento card shell for the results grid: solid white surface, 1px hairline,
 * layered soft shadow. Animates as part of the parent stagger container.
 */
const AnalysisCard = ({ title, subtitle, icon: Icon, aside, children, className = '' }) => {
  const headingId = useId();

  return (
    <motion.section
      variants={fadeUp}
      aria-labelledby={title ? headingId : undefined}
      className={`flex h-full min-w-0 flex-col rounded-2xl border border-zinc-200/70 bg-white p-6 shadow-card ${className}`}
    >
      {title && (
        <header className="mb-6 flex flex-wrap items-start justify-between gap-x-3 gap-y-2">
          {/* min width lets the aside chip wrap below instead of crushing the title in narrow columns */}
          <div className="flex min-w-[11rem] flex-1 items-center gap-3">
            {Icon && (
              <span className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl border border-zinc-200/70 bg-zinc-50 text-zinc-600">
                <Icon className="h-4 w-4" aria-hidden="true" />
              </span>
            )}
            <div className="min-w-0">
              <h4 id={headingId} className="text-[15px] font-semibold leading-5 tracking-tight text-zinc-900">
                {title}
              </h4>
              {subtitle && <p className="mt-0.5 text-xs leading-4 text-zinc-500">{subtitle}</p>}
            </div>
          </div>
          {aside && <div className="flex-shrink-0">{aside}</div>}
        </header>
      )}
      <div className="flex flex-1 flex-col">{children}</div>
    </motion.section>
  );
};

export default AnalysisCard;
