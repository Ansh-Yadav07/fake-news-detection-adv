# Fake News Detector Advanced

A production-grade fake news detection system that combines deep learning, classical machine learning, linguistic analysis, and real-time multi-source verification to deliver accurate content authenticity assessments.

## Features

### Core Capabilities
- **Parallel Analysis**: 6 concurrent verification tasks complete in ~5 seconds
- **Multi-Model Ensemble**: Deep learning + classical ML with dynamic weighting
- **Real-Time Verification**: Live cross-referencing with Wikipedia and 4 news APIs
- **Input-Aware Logic**: Adaptive analysis based on content type (fact claims vs. news articles)
- **Contradiction Detection**: Specialized Wikipedia verification for factual claims

### Verdict Categories
- **VERIFIED FACT** - Wikipedia-confirmed factual statements
- **VERIFIED** - Multi-source news confirmation
- **LIKELY REAL** - Strong supporting evidence
- **REAL** - ML models agree on authenticity
- **UNVERIFIED** - Insufficient evidence (not automatically fake)
- **SUSPICIOUS** - Mixed signals raise concerns
- **LIKELY FAKE** - Strong evidence of falsity
- **FAKE** - ML models agree on falsity

## Architecture

### System Overview
```
Input -> Classify -> 6 Parallel Tasks -> Weighted Verdict -> Output
```

### Pipeline Components

1. **Transformer Model** (Deep Learning)
   - Fine-tuned DistilBERT neural network
   - Analyzes semantic meaning and context
   - Local model with remote HuggingFace API fallback

2. **LR + TF-IDF Model** (Machine Learning)
   - Logistic Regression on TF-IDF features
   - Fast, interpretable second opinion
   - Catches patterns transformer may miss

3. **Linguistic Analysis** (NLP Features)
   - 8 handcrafted features for style analysis
   - Clickbait detection with sensationalist pattern matching
   - Punctuation density, uppercase ratio, text complexity metrics

4. **Wikipedia Verification** (Knowledge Base)
   - Entity extraction and fact verification
   - TF-IDF cosine similarity matching
   - Specialized contradiction detection for relational claims
   - Handles "X is capital of Y" type statements

5. **Multi-Source News Verification** (4 APIs)
   - GNews: Primary news source
   - NewsData.io: Secondary verification
   - NewsAPI.org: Tertiary source
   - SerpAPI (Google News): Smart fallback for insufficient results

6. **Weighted Verdict Engine** (Decision Logic)
   - Input-aware dynamic weighting
   - Fact claims: Wikipedia (60%), ML (20%), Linguistic (10%), Clickbait (10%)
   - News articles: GNews (50%), ML (20%), Source Credibility (15%), Linguistic (10%), Clickbait (5%)
   - Mixed content: Balanced combination

## Tech Stack

### Backend

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Framework | Flask | 3.0.0 | Web server & API |
| CORS | Flask-CORS | 4.0.0 | Cross-origin support |
| Deep Learning | Transformers | Latest | DistilBERT model |
| ML Framework | Scikit-learn | 1.8.0 | Classical ML models |
| Numerics | NumPy | >=1.26.4 | Numerical operations |
| Serialization | Joblib | 1.3.2 | Model persistence |
| HTTP Client | Requests | 2.31.0 | API calls |
| Production Server | Gunicorn | 21.2.0 | WSGI server |
| Model Hosting | HuggingFace Hub | 0.20.3 | Remote model access |
| Deep Learning Backend | PyTorch/Torch | Latest | Transformer backend |

### Frontend

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Framework | React | 18.2.0 | UI components |
| DOM | React DOM | 18.2.0 | DOM rendering |
| Build Tool | Vite | 5.2.0 | Fast development builds |
| Styling | Tailwind CSS | 3.4.1 | Utility-first CSS |
| CSS Preprocessor | PostCSS | 8.4.38 | CSS transformations |
| Autoprefixer | Autoprefixer | 10.4.19 | Vendor prefixes |
| Routing | React Router DOM | 6.30.3 | Navigation |
| Animations | Framer Motion | 12.38.0 | Smooth animations |
| Charts | Recharts | 3.8.1 | Data visualization |
| Icons | Lucide React | 0.363.0 | Icon library |
| Class Merging | Tailwind Merge | 3.5.0 | Class name merging |
| Utility | clsx | 2.1.1 | Conditional classes |

### Verification Sources

| Service | Type | Coverage | Timeout |
|---------|------|----------|---------|
| Wikipedia REST API | Knowledge Base | Factual claims | 1.5s |
| GNews API | News Aggregator | Real-time news | 2.0s |
| NewsData.io | News API | Alternative source | 2.0s |
| NewsAPI.org | News API | Standard source | 2.0s |
| SerpAPI (Google News) | Search Engine | Fallback | 3.0s |

## Models

### Local Models
- **Transformer**: DistilBERT fine-tuned on fake news dataset (`models/transformer/`)
- **Hybrid ML**: Logistic Regression + TF-IDF vectorizer (`models/ml_lr_model.pkl`, `models/ml_tfidf.pkl`)

### Remote Fallback
- HuggingFace Inference API: `anshy047/fake-news-detector-transformer`

## Project Structure

```
fake-news-detector-adv/
├── app.py                      # Flask backend with all API endpoints
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
├── README.md                   # This file
│
├── src/                        # Backend source code
│   ├── train_transformer.py    # Transformer training script
│   ├── train_hybrid_model.py   # Hybrid model training
│   ├── train_ml_models.py      # Baseline ML models
│   ├── preprocess.py           # Data preprocessing
│   ├── extract_embeddings.py   # Feature extraction
│   ├── linguistic_features.py  # Linguistic feature engineering
│   └── ...
│
├── models/                     # Trained models
│   ├── transformer/            # DistilBERT model files
│   ├── ml_lr_model.pkl         # Logistic Regression model
│   ├── ml_tfidf.pkl            # TF-IDF vectorizer
│   └── ...
│
├── data/                       # Datasets
│   ├── train.tsv               # Training data
│   ├── test.tsv                # Test data
│   ├── valid.tsv               # Validation data
│   └── ...
│
└── website/                    # Frontend application
    ├── src/                    # React source
    │   ├── App.jsx             # Main application
    │   ├── main.jsx            # Entry point
    │   ├── pages/              # Page components
    │   ├── components/         # UI components
    │   │   ├── analyzer/       # Analysis components
    │   │   └── ...
    │   └── utils/              # Utility functions
    │       └── decisionLogic.js # Verdict computation
    ├── package.json            # Node dependencies
    ├── tailwind.config.js      # Tailwind configuration
    ├── vite.config.js          # Vite configuration
    └── index.html              # HTML template
```

## Verdict System

### Weight Profiles by Input Type

#### Fact Claims (< 25 words, factual statements)
- Wikipedia: 60%
- ML Models: 20%
- Linguistic: 10%
- Clickbait: 10%

#### News Articles (> 25 words, event-based)
- GNews: 50%
- ML Models: 20%
- Source Credibility: 15%
- Linguistic: 10%
- Clickbait: 5%

#### Mixed Content
- Wikipedia: 30%
- GNews: 25%
- ML Models: 20%
- Linguistic: 15%
- Clickbait: 10%

### Override Rules
1. Wikipedia contradiction -> **LIKELY FAKE** (85% confidence)
2. Wikipedia score > 70% for fact claims -> **VERIFIED FACT**
3. GNews score > 50% + 2+ trusted sources -> **VERIFIED**
4. GNews score > 35% + 1+ trusted source -> **VERIFIED**
5. No evidence found -> **UNVERIFIED** (never auto-fake)

### Robustness Scoring (0-10)
- Base: Average model confidence x 10
- Bonus: +2.0 for Wikipedia score > 70%
- Bonus: +2.0 for strong GNews verification with trusted sources
- Penalty: -1.5 for model disagreement
- Penalty: -2.5 for very short text (< 10 words)

## API Endpoints

### POST /analyze
Unified analysis endpoint - runs all 6 tasks in parallel

**Request:**
```json
{
  "text": "Your text to analyze here"
}
```

**Response:**
```json
{
  "input_type": "fact_claim|news_article|mixed",
  "transformer": {
    "label": "REAL|FAKE",
    "confidence": 0.85,
    "source": "local|remote|fallback_lr"
  },
  "hybrid": {
    "label": "REAL|FAKE",
    "confidence": 0.78
  },
  "raw_features": {
    "uppercase": 0.12,
    "punctuation": 0.08,
    "clickbait": 0.25,
    "complexity": 4.5,
    "word_count": 42
  },
  "wikipedia": {
    "verification_score": 85,
    "status": "VERIFIED FACT|PARTIALLY VERIFIED|NOT FOUND|CONTRADICTED",
    "message": "...",
    "wiki_title": "...",
    "wiki_url": "...",
    "wiki_extract": "...",
    "is_contradicted": false
  },
  "verification": {
    "verification_score": 72,
    "supporting_articles": 5,
    "trusted_source_count": 3,
    "status": "VERIFIED|PARTIALLY SUPPORTED|UNVERIFIED",
    "message": "...",
    "articles": [...],
    "insights": [...]
  },
  "timings": {
    "transformer": 0.123,
    "hybrid": 0.045,
    "linguistic": 0.002,
    "wikipedia": 1.234,
    "gnews": 1.876
  },
  "total_time": 4.567
}
```

### GET /
Health check endpoint

**Response:**
```json
{
  "status": "API is running",
  "models": {
    "lr_tfidf": true,
    "transformer_local": true,
    "transformer_remote": true
  }
}
```

### POST /predict (Legacy)
Simpler endpoint with basic predictions

## Input Classification

The system automatically classifies input into one of three types:

- **fact_claim**: Short factual statements (< 25 words) with no event keywords
- **news_article**: Headlines or articles about current events with event/news keywords
- **mixed**: Uncertain cases - runs both Wikipedia and GNews verification

Classification is based on:
- Event keyword matching (politics, business, security, science, disasters, sports, legal, international)
- Proper noun detection
- Numerical patterns
- First-word acronym detection
- Text length

## Linguistic Features

The system extracts 8 features for analysis:

1. **Exclamation Count**: Number of exclamation marks
2. **Question Count**: Number of question marks
3. **Uppercase Ratio**: Proportion of uppercase characters
4. **Digit Ratio**: Proportion of digit characters
5. **Punctuation Ratio**: Proportion of punctuation characters
6. **Stopword Ratio**: Proportion of stopwords
7. **Average Word Length**: Mean word length
8. **Text Length**: Total character count

### Clickbait Detection
Computes a 0-1 score based on:
- Sensationalist trigger words (40% max)
- Excessive punctuation patterns (20% max)
- General punctuation density (10% max)
- Excessive uppercase ratio (20% max)
- ALL CAPS word patterns (25% max)
- Text brevity (10% max)

## Wikipedia Verification

### Contradiction Detection
Specialized logic for identifying false claims:
- Relational claims: "X is capital of Y" verification
- Type contradictions: "India is capital of London" (India is a country)
- Role contradictions: "X is president of Y" but Wikipedia says otherwise
- Flat Earth and well-known false claims
- Numerical contradictions

### Scoring
- TF-IDF cosine similarity: 40%
- Term overlap: 30%
- Description match boost: 30%
- Maximum score: 100%

## News Verification

### Multi-Source Strategy
1. **Phase 1**: GNews, NewsData.io, NewsAPI.org run in parallel (2s timeout)
2. **Phase 2**: If < 3 articles found, trigger SerpAPI fallback (Google News)
3. **Deduplication**: Remove duplicate articles by title similarity

### Scoring
- Weighted top matches: Best match 40%, 2nd 25%, 3rd 15%, 4th 10%, 5th 10%
- Trusted source boost: +20% max for strong trusted source matches
- Status thresholds calibrated for real-world news headlines

### Trusted Sources
Reuters, AP News, BBC, The Hindu, Indian Express, Times of India, Hindustan Times, NDTV, Associated Press, BBC News

## Setup

### Prerequisites
- Python 3.8+
- Node.js 18+
- pip
- npm

### Backend Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Fake-News-Detector-Adv
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```bash
HF_TOKEN=your_huggingface_token
GNEWS_API_KEY=your_gnews_api_key
NEWSDATA_API_KEY=your_newsdata_api_key
NEWSAPI_KEY=your_newsapi_key
SERPAPI_KEY=your_serpapi_key
```

5. Run the Flask server:
```bash
python app.py
```

Server runs on port 5001 by default.

### Frontend Setup

1. Navigate to website directory:
```bash
cd website
```

2. Install dependencies:
```bash
npm install
```

3. Configure API URL in `.env`:
```bash
VITE_API_URL=http://localhost:5001
```

4. Run development server:
```bash
npm run dev
```

Runs on port 5173 by default.

### Production Deployment

1. Build frontend:
```bash
npm run build
```

2. Deploy backend with Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

3. Serve frontend static files (from `website/dist`)

## Usage

### Local Development
1. Start backend: `python app.py`
2. Start frontend: `cd website && npm run dev`
3. Open `http://localhost:5173` in browser

### API Usage
```javascript
const response = await fetch('http://localhost:5001/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Your text to verify' })
});
const result = await response.json();
```

## Performance

- **Average Analysis Time**: ~5 seconds (all 6 tasks in parallel)
- **Model Agreement**: Measured as weighted average confidence
- **Robustness Score**: 0-10 scale based on confidence, agreement, and evidence quality
- **Verification Coverage**: 4 news APIs + Wikipedia for comprehensive coverage

## Edge Cases Handled

- Very short text (< 10 words): Linguistic features weighted heavily
- Model disagreements: Falls back to linguistic analysis
- No online evidence: Returns UNVERIFIED, not automatically fake
- Wikipedia contradictions: Immediately flags as LIKELY FAKE
- API timeouts: Graceful degradation with fallback strategies
- Local model unavailable: Falls back to HuggingFace Inference API

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - Feel free to use, modify, and distribute.

---

**Built with love using Flask, React, Transformers, and multiple verification APIs**

*Accuracy improves with longer content and available online verification sources.*
