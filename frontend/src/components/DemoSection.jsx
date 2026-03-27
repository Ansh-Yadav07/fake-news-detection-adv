import { useState } from 'react';
import { AlertTriangle, Fingerprint } from 'lucide-react';
import ResultCard from './ResultCard';

export default function DemoSection() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const analyzeText = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setResults(null);

    // Call your Flask /predict endpoint here:
    /*
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hea        hekin        hea        hase().includes("exclusive");
      const rLabel = isFake ? 'FAKE' : 'REAL';
      const conf1 = 80 + Math.random() * 15;
      const conf2 = 75 + Math.random() * 15;
      
      setResults({
        transformer: { label: rLabel, confidence: conf1, name: "DistilBERT Sequence Classifier" },
        hybrid: { label: isFake ? 'REAL' : rLabel, confidence: conf2, name: "Logistics Regression + Extracted Features" }, // Fake disagreement roughly based on mock logic
        explanations: ["High uppercase ratio", "Excessive punctuation marks (!/?)", "Contains clickbait vocabulary"]
      });
      setLoading(false);
    }, 1200);
  };

  const disagreement = results && (results.transformer.label !== results.hybrid.label);

  return (
    <section id="demo-section" className="scroll-mt-32">
      <div className="bg-white/60 backdrop-blur-xl border border-zinc-200/80 shadow-[0_8px_30px_rgb(0,0,0,0.02)] rounded-3xl p-6 md:p-10 transition-all">
        <h2 className="text-xl font-semibold mb-6 flex items-center gap-2 text-zinc-900">
          <Fingerprint className="w-5 h-5 text-zinc-400" /> Analysis Input
        </h2>
        <textarea
          className="w-full h-44 bg-zinc-50/50 border border-zinc-200 rounded-2xl p-5 text-zinc-800 focus:outline-none focus:ring-[3px] focus:ring-zinc-900/5 focus:border-zinc-400 transition-all resize-y placeholder:text-zinc-400 leading-relaxed text-base"
          placeholder="Paste news content, an article segment, or social media post here for contextual and linguistic analysis..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        
        <div className="mt-5 flex justify-end">
          <button 
            disabled={loading || !text.trim()}
            onClick={analyzeText}
            className="flex items-center gap-2 px-6 py-3 bg-zinc-900 text-white rounded-xl text-sm font-medium hover:bg-zinc-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-[2px] border-white/30 border-t-white rounded-full animate-spin" />
                Processing Base Tensors...
              </>
            ) : "Analyze Veracity"}
          </button>
        </div>

        {results && (
          <div className="mt-12 pt-10 border-t border-zinc-100 animate-fade-in">
            {disagreement && (
              <div className="mb-6 flex animate-slide-in items-start gap-4 bg-zinc-50 border border-zinc-200 rounded-2xl p-5">
                <AlertTriangle className="w-5 h-5 text-zinc-500 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-sm text-zinc-900">Disagreement Detected</h3>
                  <p className="text-sm text-zinc-500 mt-1">The Transformer and Hybrid models produced conflicting predictions. Manual fact-checking is advised for this input.</p>
                </div>
              </div>
            )}
            
            <div className="grid md:grid-cols-2 gap-6">
              <ResultCard title="Transformer Logic" data={results.transformer} />
              <ResultCard title="Hybrid Pipeline" data={results.hybrid} />
            </div>

            {results.explanations && results.explanations.length > 0 && (
              <div className="mt-6 bg-zinc-50/50 rounded-2xl p-6 border border-zinc-100">
                <h4 className="text-xs font-bold text-zinc-400 tracking-widest uppercase mb-4">Linguistic Markers</h4>
                <ul className="space-y-3">
                  {results.explanations.map((exp, i) => (
                    <li key={i} className="flex items-center text-sm text-zinc-700 gap-3 font-medium">
                      <span className="w-1.5 h-1.5 bg-zinc-800 rounded-full"></span>
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
}
