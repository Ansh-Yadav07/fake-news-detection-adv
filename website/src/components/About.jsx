import React from 'react';
import { BrainCircuit, BookOpen, Scaling } from 'lucide-react';

const About = () => {
  return (
    <section id="about" className="py-20 px-6 max-w-6xl mx-auto">
      <div className="text-center mb-16">
        <h2 className="text-3xl font-bold tracking-tight mb-4">How It Works</h2>
        <p className="text-zinc-500 max-w-xl mx-auto">
          Combining deep semantic understanding with handcrafted linguistic markers for maximum robustness.
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        <div className="glass-card p-8 group hover:bg-white/80 transition-colors">
          <div className="w-12 h-12 bg-zinc-100 rounded-xl flex items-center justify-center mb-6">
            <BrainCircuit className="w-6 h-6 text-zinc-800" />
          </div>
          <h3 className="text-xl font-semibold mb-3">Transformer Model</h3>
          <p className="text-zinc-500 text-sm leading-relaxed">
            Utilizes a fine-tuned DistilBERT neural network to analyze the deep semantic relationship and context of statements across thousands of parameters.
          </p>
        </div>

        <div className="glass-card p-8 group hover:bg-white/80 transition-colors">
          <div className="w-12 h-12 bg-zinc-100 rounded-xl flex items-center justify-center mb-6">
            <BookOpen className="w-6 h-6 text-zinc-800" />
          </div>
          <h3 className="text-xl font-semibold mb-3">Hybrid Pipeline</h3>
          <p className="text-zinc-500 text-sm leading-relaxed">
            Correlates dense BERT embeddings with 8 handcrafted statistical linguistic features—like punctuation density and capitalization ratios.
          </p>
        </div>

        <div className="glass-card p-8 group hover:bg-white/80 transition-colors">
          <div className="w-12 h-12 bg-zinc-100 rounded-xl flex items-center justify-center mb-6">
            <Scaling className="w-6 h-6 text-zinc-800" />
          </div>
          <h3 className="text-xl font-semibold mb-3">Disagreement Detection</h3>
          <p className="text-zinc-500 text-sm leading-relaxed">
            If the deep learning model and linguistic model contradict each other, the system automatically flags the text as requiring manual verification.
          </p>
        </div>
      </div>
    </section>
  );
};

export default About;