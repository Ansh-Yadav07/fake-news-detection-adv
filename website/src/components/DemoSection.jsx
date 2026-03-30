import React, { useState } from 'react';
import { ArrowRight, Loader2, Sparkles } from 'lucide-react';
import ChartSection from './analyzer/ChartSection';
import StatsGrid from './analyzer/StatsGrid';
import FeatureBars from './analyzer/FeatureBars';
import VerdictPanel from './analyzer/VerdictPanel';
import InsightsList from './analyzer/InsightsList';
import { getFinalVerdict, calculateAgreement, generateExplanations, getWeightedFusion } from '../utils/decisionLogic';

const DemoSection = () => {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleAnalyze = async () => {
    if (!input.trim()) return;
    
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch('https://fake-news-detection-5gpf.onrender.com/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input.trim() })
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const apiData = await response.json();
      
      const words = input.trim().split(/\s+/);
      const wordCount = words.length;

      // Extract from the Real API
      const t_conf = apiData.transformer.confidence;
      const t_label = apiData.transformer.label;
      const h_conf = apiData.hybrid.confidence;
      const h_label = apiData.hybrid.label;
      const rawFeatures = apiData.raw_features || {};

      const computedFeatures = {
        clickbait: rawFeatures.clickbait || 0.24,
        uppercase: rawFeatures.uppercase || 0,
        punctuation: rawFeatures.punctuation || 0,
        complexity: rawFeatures.complexity || 5
      };

      // 1. Calculate Agreement
      const agreementRaw = calculateAgreement(t_conf, h_conf, t_label, h_label);
      
      // 2. Final Verdict Logic
      let finalVerdictLabel = getFinalVerdict(t_label, t_conf, h_label, h_conf);
      if (agreementRaw > 0.7 && finalVerdictLabel === 'UNCERTAIN') {
         // Fallback if agreement is high but logic returned uncertain
         finalVerdictLabel = t_conf > h_conf ? t_label : h_label;
      }
      
      // 3. Dynamic Explanations
      const explanations = generateExplanations(computedFeatures, t_label, h_label, finalVerdictLabel);

      const dashboardData = {
        transformer: { 
          label: t_label, 
          confidence: t_conf * 100 // mapped to percentages for UI display
        },
        hybrid: { 
          label: h_label, 
          confidence: h_conf * 100 
        },
        stats: {
          wordCount: wordCount,
          avgWordLength: (input.length / Math.max(1, wordCount)).toFixed(1),
          upperRatio: Math.round(computedFeatures.uppercase * 100),
          punctDensity: Math.round(computedFeatures.punctuation * 100)
        },
        features: {
          punctuation: Math.round(computedFeatures.punctuation * 100), // scale to 100 for UI bar
          uppercase: Math.round(computedFeatures.uppercase * 100),
          complexity: computedFeatures.complexity, // remains 0-10
          clickbait: Math.round(computedFeatures.clickbait * 100)
        },
        agreementScore: Math.round(agreementRaw * 100),
        robustnessScore: 8.5,
        finalVerdictLabel: finalVerdictLabel,
        verdictExplanation: explanations.length > 0 ? explanations[0] : "Analysis complete.",
        explanations: explanations
      };
      
      setResult(dashboardData);
    } catch (err) {
      console.error(err);
      alert("Failed to connect to the backend API.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="demo" className="py-10 px-6 max-w-6xl mx-auto">
      <div className="text-center mb-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 mb-6 rounded-full bg-zinc-100 border border-zinc-200 text-xs font-semibold tracking-wide text-zinc-600 shadow-sm">
          <Sparkles className="w-4 h-4" />
          Intelligence Dashboard
        </div>
        <h2 className="text-3xl md:text-5xl font-black tracking-tight mb-4 text-zinc-900">
          Content Analysis Engine
        </h2>
        <p className="text-zinc-500 max-w-xl mx-auto text-sm md:text-base font-medium">
          Paste your article or headline below. Our dual-model pipeline will process linguistic markers, analyze syntactic complexity, and establish a multi-layered truthfulness verdict.
        </p>
      </div>

      <div className="glass-card p-6 md:p-8 rounded-3xl bg-white/70 shadow-xl border border-black/5 ring-1 ring-black/5 mb-12 mx-auto max-w-4xl relative z-10 transition-all">
        <div className="relative mb-6">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Paste text here... (e.g. 'BREAKING: Secret documents exposed!')"
            className="w-full min-h-[160px] p-6 rounded-2xl bg-white/80 border border-zinc-200 text-zinc-900 placeholder:text-zinc-400 focus:outline-none focus:ring-4 focus:ring-zinc-900/5 focus:border-zinc-300 focus:bg-white transition-all resize-y text-base md:text-lg font-medium shadow-inner"
          />
        </div>

        <div className="flex justify-end">
          <button 
            onClick={handleAnalyze} 
            disabled={loading || !input.trim()}
            className="w-full md:w-auto px-10 py-4 bg-zinc-900 text-white rounded-xl font-bold tracking-wide hover:bg-zinc-800 focus:ring-4 focus:ring-zinc-900/20 transition-all shadow-lg hover:-translate-y-0.5 disabled:opacity-50 disabled:hover:translate-y-0 flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Processing Signal...
              </>
            ) : (
              <>
                Initialize Analysis
                <ArrowRight className="w-4 h-4 ml-1" />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Modern Analytics Dashboard Render */}
      {result && (
        <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out space-y-6">
          <VerdictPanel 
            label={result.finalVerdictLabel}
            transformerLabel={result.transformer.label}
            hybridLabel={result.hybrid.label}
            explanation={result.verdictExplanation}
            agreementScore={result.agreementScore}
            robustnessScore={result.robustnessScore}
          />

          <div className="grid md:grid-cols-3 gap-6">
            <ChartSection transformerConf={result.transformer.confidence} hybridConf={result.hybrid.confidence} />
            <FeatureBars {...result.features} />
            <StatsGrid {...result.stats} />
          </div>

          <div className="grid grid-cols-1">
            <InsightsList explanations={result.explanations} />
          </div>
        </div>
      )}
    </section>
  );
};

export default DemoSection;