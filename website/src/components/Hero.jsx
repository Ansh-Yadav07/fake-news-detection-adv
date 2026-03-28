import React from 'react';
import { ShieldCheck } from 'lucide-react';

const Hero = () => {
  return (
    <section className="pt-6 pb-0 px-6 max-w-5xl mx-auto text-center flex flex-col items-center justify-center animate-in fade-in zoom-in duration-700">
      <div className="inline-flex items-center gap-4 px-3 py-1.5 mb-8 rounded-full bg-zinc-100 border border-zinc-200 text-xs font-semibold tracking-wide text-zinc-600 shadow-sm transition-all hover:bg-zinc-200">
        <ShieldCheck className="w-4 h-4" />
        Advanced System Architecture
      </div>
      
      <h1 className="text-5xl md:text-7xl font-extrabold tracking-tighter mb-6 text-zinc-900 leading-[1.1]">
        AI-Powered <br className="hidden md:block" />
        Fake News Detection
      </h1>
      
      <p className="text-lg md:text-xl text-zinc-500 mb-10 max-w-xl mx-auto leading-relaxed">
        Verify the integrity of any article instantly. Our dual-engine approach uses state-of-the-art Transformers and linguistic Hybrid models to evaluate truthfulness with high confidence.
      </p>
    </section>
  );
};

export default Hero;