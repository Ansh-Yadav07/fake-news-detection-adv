import React, { useState } from 'react';
import { ArrowRight, Loader2, Sparkles, WifiOff, Zap, BookOpen, Newspaper } from 'lucide-react';
import ChartSection from './analyzer/ChartSection';
import StatsGrid from './analyzer/StatsGrid';
import FeatureBars from './analyzer/FeatureBars';
import VerdictPanel from './analyzer/VerdictPanel';
import InsightsList from './analyzer/InsightsList';
import OnlineVerification from './analyzer/OnlineVerification';
import { getEnhancedDecision, getEnhancedDecisionWithVerification, calculateRobustness, calculateRobustnessWithVerification } from '../utils/decisionLogic';

const INPUT_TYPE_CONFIG = {
  fact_claim: { label: 'Fact Claim', icon: BookOpen, color: 'text-violet-700 bg-violet-50 border-violet-200' },
  news_article: { label: 'News Article', icon: Newspaper, color: 'text-sky-700 bg-sky-50 border-sky-200' },
  mixed: { label: 'Mixed Content', icon: Zap, color: 'text-amber-700 bg-amber-50 border-amber-200' },
};

const DemoSection = () => {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [loadingPhase, setLoadingPhase] = useState('');

  const handleAnalyze = async () => {
    if (!input.trim()) return;
    
    setLoading(true);
    setResult(null);
    setLoadingPhase('Initializing parallel analysis...');

    try {
      const apiBase = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:5001/predict').replace('/predict', '');
      const textPayload = JSON.stringify({ text: input.trim() });

      // Single unified call — all 6 tasks run in parallel on the backend
      setLoadingPhase('Running 6 parallel tasks...');
      
      const response = await fetch(`${apiBase}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: textPayload
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const apiData = await response.json();
      if (apiData.error) {
        throw new Error(apiData.error);
      }

      setLoadingPhase('Computing verdict...');

      const words = input.trim().split(/\s+/);
      const wordCount = words.length;

      // Extract from unified response
      const t_conf = apiData.transformer.confidence;
      const t_label = apiData.transformer.label;
      const h_conf = apiData.hybrid.confidence;
      const h_label = apiData.hybrid.label;
      const rawFeatures = apiData.raw_features || {};
      const inputType = apiData.input_type || 'mixed';
      const wikipedia = apiData.wikipedia || null;
      const verification = apiData.verification || null;

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

      // Determine if ANY verification source is available
      const wikiAvailable = wikipedia && wikipedia.status && wikipedia.status !== "NOT FOUND";
      const gnewsAvailable = verification && !verification.error && (
        (verification.articles && verification.articles.length > 0) || 
        verification.supporting_articles > 0 ||
        verification.verification_score > 0
      );
      const hasVerification = wikiAvailable || gnewsAvailable;

      // Get decision — WITH or WITHOUT verification
      let decision;
      let robustnessScore;

      if (hasVerification) {
        decision = getEnhancedDecisionWithVerification(
          t_label, t_conf,
          h_label, h_conf,
          clickbaitScore, pctScore, uppercaseScore,
          wordCount,
          inputType,
          wikipedia,
          verification
        );
        robustnessScore = calculateRobustnessWithVerification(
          t_conf, h_conf, decision.agreement / 100, wordCount, wikipedia, verification
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
        inputType: inputType,
        wikipedia: wikiAvailable ? wikipedia : null,
        verification: gnewsAvailable ? verification : null,
        verificationSource: decision.verification_source || 'none',
        verificationWeights: decision.weights || null,
        verificationFailed: !hasVerification,
        timings: apiData.timings || null,
        totalTime: apiData.total_time || null
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

  const inputTypeConfig = result ? INPUT_TYPE_CONFIG[result.inputType] || INPUT_TYPE_CONFIG.mixed : null;

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
          Paste your article or headline below. Our parallel verification engine runs 6 concurrent checks — ML models, linguistic analysis, Wikipedia facts, and live news sources.
        </p>
      </div>

      <div className="glass-card p-6 md:p-8 rounded-3xl bg-white/70 shadow-xl border border-black/5 ring-1 ring-black/5 mb-12 mx-auto max-w-4xl relative z-10 transition-all">
        <div className="relative mb-6">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Paste text here... "
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

      {/* Analytics Dashboard */}
      {result && (
        <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out space-y-6">
          {/* Input Type Badge + Timing */}
          <div className="flex items-center justify-between">
            {inputTypeConfig && (
              <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-bold uppercase tracking-wide ${inputTypeConfig.color}`}>
                <inputTypeConfig.icon className="w-3.5 h-3.5" />
                {inputTypeConfig.label}
              </div>
            )}
            {result.totalTime && (
              <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-50 border border-zinc-200 text-xs font-semibold text-zinc-500">
                <Zap className="w-3.5 h-3.5" />
                {result.totalTime}s — 6 parallel tasks
              </div>
            )}
          </div>

          <VerdictPanel 
            label={result.finalVerdictLabel}
            transformerLabel={result.transformer.label}
            hybridLabel={result.hybrid.label}
            explanation={result.verdictExplanation}
            agreementScore={result.agreementScore}
            robustnessScore={result.robustnessScore}
            verificationWeights={result.verificationWeights}
            inputType={result.inputType}
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
            <OnlineVerification 
              wikipedia={result.wikipedia}
              verification={result.verification}
              verificationSource={result.verificationSource}
              inputType={result.inputType}
            />
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