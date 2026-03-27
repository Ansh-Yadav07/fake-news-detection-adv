import React, { useState } from 'react';
import { AlertCircle, ArrowRight, Loader2 } from 'lucide-react';

const DemoSection = () => {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleAnalyze = async () => {
    if (!input.trim()) return;
    
    setLoading(true);
    setResult(null);

    try {
      // TODO: Replace with proper Fetch to your Flask backend:
      // const res = await fetch('/predict', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ text: input })
      // });
      // const data = await res.json();
      
      // Mocked latency response
      await new Promise(r => setTimeout(r, 1800));
      const mockedData = {
        transformer: { label: input.length % 2 === 0 ? 'REAL' : 'FAKE', confidence: 94.2 },
        hybrid: { label: 'FAKE', confidence: 81.5 },
        explanations: [
          "High punctuation (exclamation marks) -> possible fake/clickbait pattern",
          "High uppercase ratio -> common in sensationalism"
        ]
      };
      
      setResult(mockedData);
    } catch (err) {
      console.error(err);
      alert("Failed to connect to the backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="demo" className="py-20 px-6 max-w-4xl mx-auto">
      <div className="glass-card p-6 md:p-10 text-left">
        <h3 className="text-2xl font-bold tracking-tight mb-2">Live Analysis</h3>
        <p className="text-zinc-500 text-sm mb-6">Enter an article snippet or headline below to evaluate.</p>
        
        <div className="mb-6 relative">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Paste text here... (e.g. 'BREAKING: Secret documents exposed!')"
            className="w-full min-h-[160px] p-5 rounded-2xl bg-white/50 border border-black/10 text-zinc-900 placeholder:text-zinc-400 focus:outline-none focus:ring-4 focus:ring-black/5 focus:border-black/20 focus:bg-white transition-all resize-y text-base font-medium"
          />
        </div>

        <button 
          onClick={handleAnalyze} 
          disabled={loading || !input.trim()}
          className="interactive-btn w-full md:w-auto px-8"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              Run Detection
              <ArrowRight className="w-4 h-4 ml-1" />
            </>
          )}
        </button>

        {/* Results Render */}
        {result && (
          <div className="mt-10 pt-8 border-t border-black/10 animate-fade-in">
            {result.transformer.label !== result.hybrid.label && (
              <div className="mb-8 p-4 rounded-xl bg-zinc-100 border-l-4 border-zinc-500 flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-zinc-600 flex-shrink-0" />
                <p className="text-sm font-medium text-zinc-800">
                  Model Disagreement Detected. The semantic interpretation contradicts the linguistic markers. Manual review advised.
                </p>
              </div>
            )}

            <div className="grid md:grid-cols-2 gap-6 mb-8">
              {/* Transformer Card */}
              <div className="bg-white/60 border border-black/5 rounded-2xl p-6">
                <div className="flex justify-between items-center mb-6">
                  <span className="text-xs font-bold tracking-wider text-zinc-500 uppercase">Transformer</span>
                  <span className={`px-3 py-1 rounded-full text-xs font-bold border ${result.transformer.label === 'REAL' ? 'border-zinc-300 bg-zinc-50 text-zinc-900' : 'border-zinc-900 bg-zinc-900 text-white'}`}>
                    {result.transformer.label}
                  </span>
                </div>
                <div className="text-3xl font-bold tracking-tight mb-1">{result.transformer.confidence.toFixed(1)}%</div>
                <div className="text-xs text-zinc-500 mb-4">Confidence Score</div>
                <div className="h-1.5 w-full bg-black/5 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-zinc-900 rounded-full progress-bar-transition"
                    style={{ width: `${result.transformer.confidence}%` }}
                  />
                </div>
              </div>

              {/* Hybrid Card */}
              <div className="bg-white/60 border border-black/5 rounded-2xl p-6">
                <div className="flex justify-between items-center mb-6">
                  <span className="text-xs font-bold tracking-wider text-zinc-500 uppercase">Hybrid Model</span>
                  <span className={`px-3 py-1 rounded-full text-xs font-bold border ${result.hybrid.label === 'REAL' ? 'border-zinc-300 bg-zinc-50 text-zinc-900' : 'border-zinc-900 bg-zinc-900 text-white'}`}>
                    {result.hybrid.label}
                  </span>
                </div>
                <div className="text-3xl font-bold tracking-tight mb-1">{result.hybrid.confidence.toFixed(1)}%</div>
                <div className="text-xs text-zinc-500 mb-4">Confidence Score</div>
                <div className="h-1.5 w-full bg-black/5 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-zinc-900 rounded-full progress-bar-transition"
                    style={{ width: `${result.hybrid.confidence}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Explanation Area */}
            {result.explanations && result.explanations.length > 0 && (
              <div className="bg-white/40 border border-black/5 p-6 rounded-2xl">
                <h4 className="text-sm font-bold text-zinc-900 mb-4">Linguistic Markers & Explanations</h4>
                <ul className="space-y-3">
                  {result.explanations.map((exp, idx) => (
                    <li key={idx} className="flex items-start gap-3 text-sm text-zinc-600">
                      <div className="mt-1 w-1.5 h-1.5 rounded-full bg-zinc-400 flex-shrink-0" />
                      {exp}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
};

export default DemoSection;