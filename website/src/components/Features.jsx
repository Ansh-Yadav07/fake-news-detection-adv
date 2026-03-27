import React from 'react';
import { Target, Activity, ShieldAlert, Cpu, Sparkles } from 'lucide-react';

const Features = () => {
  const featuresList = [
    {
      title: "Dual-Model Prediction",
      desc: "Scores are cross-referenced between two independently trained architectures.",
      icon: <Target className="w-5 h-5" />
    },
    {
      title: "Confidence Scoring",
      desc: "Granular probability outputs mapped distinctly as a percentage layout.",
      icon: <Activity className="w-5 h-5" />
    },
    {
      title: "Self-Reflection Safety",
      desc: "Detects edge-case uncertainty safely logging UNCERTAIN states when below 60% confidence.",
      icon: <ShieldAlert className="w-5 h-5" />
    },
    {
      title: "Robustness Testing",
      desc: "Behind the scenes, we perturb your input (caps, punct variations) to verify stability.",
      icon: <Cpu className="w-5 h-5" />
    },
    {
      title: "Linguistic Explainability",
      desc: "Clear English text highlighting sentence factors (e.g. 'High uppercase ratio').",
      icon: <Sparkles className="w-5 h-5" />
    }
  ];

  return (
    <section id="features" className="py-20 px-6 max-w-4xl mx-auto border-t border-black/5">
      <div className="text-center mb-12">
        <h2 className="text-3xl font-bold tracking-tight">System Features</h2>
      </div>
      
      <div className="flex flex-col gap-4">
        {featuresList.map((feature, i) => (
          <div key={i} className="flex items-start gap-4 p-5 rounded-2xl hover:bg-black/[0.02] transition-colors border border-transparent hover:border-black/5">
            <div className="mt-1 p-2 bg-white border border-black/5 rounded-lg shadow-sm">
              {feature.icon}
            </div>
            <div>
              <h4 className="text-base font-semibold text-zinc-900">{feature.title}</h4>
              <p className="text-sm text-zinc-500 mt-1">{feature.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};

export default Features;