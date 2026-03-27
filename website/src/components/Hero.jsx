import React from 'react';
import { ShieldCheck } from 'lucide-react';

const Hero = () => {
  return (
    <section className="pt-32 pb-20 px-6 max-w-4xl mx-auto text-center animate-fade-in">
      <div className="inline-flex items-center gap-2 px-3 py-1 mb-6 rounded-full bg-zinc-200/50 border border-zinc-200 text-xs font-medium text-zinc-600">
        <ShieldCheck className="w-4 h-4" />
        Advanced System Architecture
      </div>
      <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6 text-zinc-900 leading-tight">
        AI-Powered <br className="hidden md:block" />
        Fake News Detection
      </h1>
      <p className="text-lg md:text-xl text-zinc-500 mb-10 max-w-2xl mx-auto">
        Verify the integrity of any article instantly. Our dual-engine approach uses state-of-the-art Transformers and linguistic Hybrid models to evaluate truthfulness with high confidence.
      </p>
      <div className="flex justify-center">
        <a href="#demo" className="interactive-btn text-lg px-8">
          Try Demo
        </a>
      </div>
    </section>
  );
};

export default Hero;