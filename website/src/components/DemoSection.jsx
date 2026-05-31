import React, { useState } from 'react';
import { ArrowRight, Loader2, Sparkles, Globe, Wifi, WifiOff } from 'lucide-react';
import ChartSection from './analyzer/ChartSection';
import StatsGrid from './analyzer/StatsGrid';
import FeatureBars from './analyzer/FeatureBars';
import VerdictPanel from './analyzer/VerdictPanel';
import InsightsList from './analyzer/InsightsList';
import OnlineVerification from './analyzer/OnlineVerification';
import { getEnhancedDecision, getEnhancedDecisionWithVerification, calculateRobustness, calculateRobustnessWithVerification } from '../utils/decisionLogic';

const DemoSection = () => {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [loadingPhase, setLoadingPhase] = useState('');

  const handleAnalyze = async () => {
    if (!input.trim()) return;
    
    setLoading(true);
    setResult(null);
    setLoadingPhase('Initializing analysis...');

    try {
      const apiBase = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:5001/predict').replace('/predict', '');
      const textPayload = JSON.stringify({ text: input.trim() });

      // Fire BOTH API calls in parallel for best performance
      setLoadingPhase('Running ML models & verifying sources...');
      
      const [predictResult, verifyResult] = await Promise.allSettled([
        fetch(`${apiBase}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: textPayload
        }).then(async (res) => {
          if (!res.ok) throw new Error(`API error: ${res.status}`);
          const data = await res.json();
          if (data.error) throw new Error(data.error);
          return data;
        }),
        fetch(`${apiBase}/verify`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: textPayload
        }).then(async (res) => {
          if (!res.ok) throw new Error(`Verify API error: ${res.status}`);
          return await res.json();
        })
      ]);

      // ML prediction is required
      if (predictResult.status === 'rejected') {
        throw new Error(predictResult.reason?.message || 'Failed to connect to ML prediction API');
      }

      const apiData = predictResult.value;

      // Verification is optional — graceful fallback
      const verificationData = verifyResult.status === 'fulfilled' ? verifyResult.value : null;
      const verificationAvailable = verificationData && !verificationData.error;

      setLoadingPhase('Computing verdict...');

      const words = input.trim().split(/\s+/);
      const wordCount = words.length;

      // Extract from the API response
      const t_conf = apiData.transformer.confidence;
      const t_label = apiData.transformer.label;
      const h_conf = apiData.hybrid.confidence;
      const h_label = apiData.hybrid.label;
      const rawFeatures = apiData.raw_features || {};

      const computedFeatures = {
        clickbait: rawFeatures.clickbait || 0,
        uppercase: rawFeatures.uppercase || 0,
        punctuation: rawFeatures.punctuation || 0,
        complexity: rawFeatures.complexity || 5
      };

      // Scale features for decision logic
      const clickbaitScore = Math.round(computedFeatures.clickbait * 100);
      const uppercaseScore = Math.round(computedFeatures.uppercase * 100);
      const pctScore = Math.round(computedFeatures.punctuation * 100);

      // Get decision — WITH or WITHOUT verification
      let decision;
      let robustnessScore;

      if (verificationAvailable) {
        decision = getEnhancedDecisionWithVerification(
          t_label, t_conf,
          h_label, h_conf,
          clickbaitScore, pctScore, uppercaseScore,
          wordCount,
          verificationData
        );
        robustnessScore = calculateRobustnessWithVerification(
          t_conf, h_conf, decision.agreement / 100, wordCount, verificationData
        );
      } else {
        decision = getEnhancedDecision(
          t_label, t_conf,
          h_label, h_conf,
          clickbaitScore, pctScore, uppercaseScore,
          wordCount
        );
        robustnessScore = calculateRobustness(
          t_conf, h_conf, decision.agreement / 100, wordCount
        );
      }
      
      const finalVerdictLabel = decision.final_label;
      const agreementScore = decision.agreement;
      
      // Use the rich insights from the decision logic
      const explanations = decision.insights || [decision.reason];

      const dashboardData = {
        transformer: { 
          label: t_label, 
          confidence: t_conf * 100,
          source: apiData.transformer.source || 'local'
        },
        hybrid: { 
          label: h_label, 
          confidence: h_conf * 100 
        },
        stats: {
          wordCount: wordCount,
          avgWordLength: (input.length / Math.max(1, wordCount)).toFixed(1),
          upperRatio: uppercaseScore,
          punctDensity: pctScore
        },
        features: {
          punctuation: pctScore,
          uppercase: uppercaseScore,
          complexity: computedFeatures.complexity,
          clickbait: clickbaitScore
        },
        agreementScore: agreementScore,
        robustnessScore: robustnessScore,
        finalVerdictLabel: finalVerdictLabel,
        verdictExplanation: decision.reason,
        explanations: explanations,
        verification: verificationAvailable ? verificationData : null,
        verificationWeights: decision.weights || null,
        verificationFailed: !verificationAvailable
      };
      
      setResult(dashboardData);
    } catch (err) {
      console.error(err);
      alert("Failed to connect to the backend API. Make sure the server is running.");
    } finally {
      setLoading(false);
      setLoadingPhase('');
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
          Paste your article or headline below. Our dual-model pipeline will process linguistic markers, verify against trusted news sources, and establish a multi-layered truthfulness verdict.
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
                {loadingPhase || 'Processing Signal...'}
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
            verificationWeights={result.verificationWeights}
          />

          {/* Verification status banner */}
          {result.verificationFailed && (
            <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-amber-50 border border-amber-200 text-amber-800">
              <WifiOff className="w-4 h-4 flex-shrink-0" />
              <span className="text-xs font-semibold">Online verification unavailable — verdict based on ML models and linguistic analysis only.</span>
            </div>
          )}

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <ChartSection transformerConf={result.transformer.confidence} hybridConf={result.hybrid.confidence} />
            <FeatureBars {...result.features} />
            <StatsGrid {...result.stats} />
            <OnlineVerification verification={result.verification} />
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