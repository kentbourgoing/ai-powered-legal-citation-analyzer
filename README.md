# AI-Powered Legal Citation Analyzer

An explainable AI legal research platform that helps attorneys assess whether a case remains "Good Law" by analyzing citation treatment across thousands of ADA decisions with configurable, transparent reasoning.

---

## Problem and Goal

- **Problem:** Traditional legal citators (Shepard's, KeyCite) are proprietary black boxes that don't explain how they classify case law strength. Attorneys spend hours manually reviewing citations to understand if precedents remain valid.
- **Why It Matters:** Legal professionals need transparent, customizable tools to assess case validity efficiently. Explainability builds trust and allows attorneys to tune analysis to their specific practice areas and risk tolerance.
- **Goal:** Build an MVP that matches traditional citator accuracy while providing paragraph-level rationales and user-configurable scoring parameters, delivered through a 14-week client capstone with Wolters Kluwer.

---

## Demo and Screenshots

### Application Homepage
![Application Homepage](UI%20Images/homepage.png)

Users can switch between Case Lookup mode for targeted search and citation analysis, or Chatbot mode for natural language queries about ADA cases, their citation history, and legal concepts stored in the Neo4j knowledge graph.

### Case Detail View - Guy Amir v. St. Louis University
![Case Detail View](UI%20Images/case-detail-view.png)

Case Lookup example: The system evaluates "Guy Amir v. St. Louis University" as "Good" law based on 46 incoming citations, primarily from Courts of Appeals. The detailed rationale explains how time-weighted citation analysis and court hierarchy inform the final label.

### Configuration Panel
![Configuration Panel](UI%20Images/configuration-panel.png)

The application allows users to configure case labeling parameters, including proportion thresholds, time-based weighting, and court hierarchy strategies to customize analysis for different practice areas.

### Chatbot Interface - Citation Analysis
![Chatbot Interface](UI%20Images/chatbot-interface.png)

Chatbot demonstration: This example shows the system identifying cases that criticize "Access Now, Inc. v. Southwest Airlines Co." and explaining how courts narrowed its holding, illustrating the platform's ability to deliver transparent, paragraph-level legal reasoning.

---

## What I Delivered

- **Dual-Mode Streamlit Web Application:** Case lookup interface for targeted citation analysis + AI chatbot for natural language legal research
- **Neo4j Legal Knowledge Graph:** 3,500+ case nodes, 5,500+ citation edges with ADA decisions, opinions, and metadata
- **Ensemble Citation Classifier:** 3-model LLM pipeline (Claude 3.5 Sonnet, Mistral-7B, Llama 3-70B) with majority voting for treatment classification
- **Configurable Case Labeling Algorithm:** Time-weighted scoring system that aggregates citation signals across court levels with adjustable parameters
- **Agentic GraphRAG Chatbot:** LangGraph-orchestrated agent with 8+ tools for semantic search, text-to-Cypher queries, and case recommendations
- **Client Deliverables:** Three milestone presentations to Wolters Kluwer stakeholders demonstrating MVP functionality and business value

---

## Results

- **Citation Classification Accuracy:** 67% accuracy, 70% precision on treatment classification (Positive/Negative/Neutral) using ensemble voting
- **Knowledge Graph Scale:** 3,500 case nodes, 5,500 citation edges extracted from ADAH handbook and CourtListener API
- **Explainability:** 100% of classifications include LLM-generated paragraph-level rationales explaining citation treatment context
- **User Configurability:** 6+ adjustable parameters (proportion thresholds, time weighting, jurisdiction weights, court hierarchy strategy)
- **Retrieval Performance:** GraphRAG system with vector similarity search (1024-dim Titan embeddings, cosine similarity) over chunked opinion text
- **Client Engagement:** Delivered on schedule through 14-week capstone with weekly stakeholder meetings and three formal milestone presentations

---

## Approach

1. **Knowledge Graph Construction:** Extracted 3,500+ ADA cases from ADAH PDF and CourtListener API, normalized case names, built ETL pipeline to construct Neo4j graph with cases, citations, courts, jurisdictions, and opinion chunks
2. **Opinion Summarization:** Generated single-paragraph case summaries using Mistral-7B via Amazon Bedrock with token-based chunking for long opinions
3. **Citation Snippet Extraction:** Developed pattern-matching pipeline to locate citation references in opinion text, extract context windows (snippets), handle "Id." references, and merge overlapping snippets
4. **Ensemble Citation Classification:** Built 3-model LLM ensemble (Claude, Mistral, Llama) with majority voting and prompt engineering to classify citation treatment (Positive/Negative/Neutral/Unknown) and generate rationales
5. **Case Labeling Algorithm:** Designed time-weighted aggregation system that groups incoming citations by court level, applies recency/jurisdiction weights, computes treatment proportions, and assigns Good/Bad/Moderate/Unknown labels using configurable thresholds and "walk down" strategy across court hierarchy
6. **GraphRAG System:** Created vector embeddings (Amazon Titan) for all opinion chunks, built Neo4j vector index, developed LangGraph agent with 8+ tools (semantic search, text-to-Cypher, recommendations, filtering) for natural language queries
7. **UI Development:** Built dual-mode Streamlit application with case lookup (search, detail view, citation analysis, CSV export) and chatbot (GraphRAG-powered Q&A) with real-time parameter configuration
8. **Evaluation and Iteration:** Conducted comparative evaluation against ground truth labels, performed error analysis, iterated on prompts and thresholds based on stakeholder feedback

---

## Tech/Methods

**Languages & Frameworks:** Python, Streamlit, LangChain, LangGraph, Neo4j Cypher

**LLMs & APIs:** Amazon Bedrock (Claude 3.5 Sonnet, Mistral-7B, Llama 3-70B), Amazon Titan Embeddings (1024-dim), CourtListener API

**Infrastructure & Data:** AWS (Bedrock, SageMaker Studio, S3), Neo4j Aura (graph database + vector index)

**Methods:** GraphRAG, Retrieval-Augmented Generation (RAG), Ensemble ML (majority voting), Agentic AI (LangGraph orchestration), Vector similarity search (cosine), Text-to-Cypher, Prompt engineering, Token-based text chunking

---

## Repo Structure

```
capstone-project-legal-citation/
├── Knowledge Graph/              # Step 1: Build Neo4j knowledge graph
│   ├── ADAH_API_Extract.ipynb           # Extract case names, fetch CourtListener data
│   ├── JSON_to_CSV_Converter.ipynb      # Format for Neo4j import
│   ├── Summarization_Pipeline.ipynb     # Generate case summaries (Mistral-7B)
│   └── EDA.ipynb                        # Exploratory data analysis
│
├── Citation Classifier/          # Step 2: Classify citation edges
│   ├── Snippet_Retriever.ipynb          # Extract citation context snippets
│   ├── Edge_Classifier_Snippet_Method_Ensemble.ipynb  # Ensemble voting (BEST)
│   ├── Edge_Classifier_Snippet_Method_Claude.ipynb
│   ├── Edge_Classifier_Snippet_Method_Llama3.ipynb
│   ├── Edge_Classifier_Snippet_Method_Mistral.ipynb
│   └── evaluation_pipeline.ipynb        # Evaluate classifier performance
│
├── Case Classifier/              # Step 3: Label cases (Good/Bad/Moderate)
│   ├── Case_Labeler.ipynb               # Run case labeling workflow
│   └── case_labeler.py                  # Time-weighted aggregation algorithm
│
├── UI/                           # Step 4: Streamlit web app + GraphRAG
│   ├── app.py                           # Main Streamlit application (ENTRY POINT)
│   ├── graph_rag_agents.py              # LangGraph agents + tools
│   ├── embed_opinion_chunks.ipynb       # Create Titan embeddings
│   ├── GraphRAG_and_Agents.ipynb        # Test GraphRAG agents
│   └── run_app.ipynb                    # Launch app from notebook
│
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
└── README.md
```

---

## How to Run (Local)

### Prerequisites
- **Python 3.9+**
- **Neo4j Database** (local or cloud instance)
- **AWS Account** with Bedrock access (Claude, Mistral, Llama, Titan models enabled)
- **CourtListener API Key** (optional, for data extraction)

### Install Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kentbourgoing/ai-powered-legal-citation-analyzer.git
   cd ai-powered-legal-citation-analyzer
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```env
   # Neo4j Configuration
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   NEO4J_DATABASE=neo4j

   # AWS Configuration
   AWS_REGION=us-west-2
   CLAUDE_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
   TITAN_MODEL_ID=amazon.titan-embed-text-v2:0

   # CourtListener API (optional, for data extraction)
   COURTLISTENER_API_KEY=your_api_key
   ```

### Commands to Run

**Build the knowledge graph (one-time setup):**
```bash
# 1. Extract cases and build CSV files
jupyter notebook "Knowledge Graph/ADAH_API_Extract.ipynb"
jupyter notebook "Knowledge Graph/JSON_to_CSV_Converter.ipynb"

# 2. Manually upload CSV files to Neo4j Aura using Data Importer
#    (follow instructions in JSON_to_CSV_Converter.ipynb)

# 3. Generate case summaries
jupyter notebook "Knowledge Graph/Summarization_Pipeline.ipynb"
```

**Classify citations (one-time setup):**
```bash
# 1. Extract citation snippets
jupyter notebook "Citation Classifier/Snippet_Retriever.ipynb"

# 2. Run ensemble classifier (RECOMMENDED)
jupyter notebook "Citation Classifier/Edge_Classifier_Snippet_Method_Ensemble.ipynb"
```

**Label cases (one-time setup):**
```bash
jupyter notebook "Case Classifier/Case_Labeler.ipynb"
```

**Embed opinion chunks and launch app:**
```bash
# 1. Create embeddings
jupyter notebook "UI/embed_opinion_chunks.ipynb"

# 2. Create Neo4j vector index (run in Neo4j Browser)
# CREATE VECTOR INDEX chunkEmbeddings IF NOT EXISTS
# FOR (c:OpinionChunk) ON c.embedding
# OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}

# 3. Launch Streamlit app
cd UI
streamlit run app.py
```

### Example Inputs and Expected Outputs

**Case Lookup Mode:**
- **Input:** Search for "Guy Amir v. St. Louis University"
- **Output:** Case detail page showing:
  - Case metadata (name, citation, date, court)
  - AI-generated summary
  - Label: "Good Law" (46 incoming citations, primarily positive from Courts of Appeals)
  - Detailed rationale explaining time-weighted citation analysis
  - Table of citing cases with treatment labels and rationales
  - CSV export option

**Chatbot Mode:**
- **Input:** "What cases discuss reasonable accommodation in the workplace?"
- **Output:** AI agent response with:
  - List of relevant cases with summaries
  - Citation context from opinion text
  - Treatment analysis (how later courts cited these cases)

---

## How to Reproduce Results

### Citation Classification Evaluation

**Command:**
```bash
jupyter notebook "Citation Classifier/evaluation_pipeline.ipynb"
```

**What it does:**
- Compares ensemble model predictions against ground truth labels
- Calculates accuracy, precision, recall, F1 for each treatment class
- Generates confusion matrix and classification report

**Metrics come from:**
- Evaluation script: `Citation Classifier/evaluation_pipeline.ipynb`
- Ground truth labels must be manually created (sample set of ~100 citations)
- Reported metrics: 67% accuracy, 70% precision on 3-class classification (Positive/Negative/Neutral)

**Seed info:**
- LLM calls use temperature=0 for reproducibility (Claude, Mistral, Llama)
- Random seeds not applicable (deterministic LLM inference)

---

## Configuration

### Environment Variables

**Neo4j:**
- `NEO4J_URI` - Database connection string (e.g., `bolt://localhost:7687`)
- `NEO4J_USERNAME` - Database username (default: `neo4j`)
- `NEO4J_PASSWORD` - Database password
- `NEO4J_DATABASE` - Database name (default: `neo4j`)

**AWS Bedrock:**
- `AWS_REGION` - AWS region (e.g., `us-west-2`)
- `CLAUDE_MODEL_ID` - Claude model ID (e.g., `anthropic.claude-3-5-sonnet-20240620-v1:0`)
- `TITAN_MODEL_ID` - Titan embeddings model ID (e.g., `amazon.titan-embed-text-v2:0`)

**CourtListener API:**
- `COURTLISTENER_API_KEY` - API key for case data extraction (optional)

### Case Labeling Configuration

The case labeling algorithm supports extensive tuning via `Case Classifier/case_labeler.py`:

**Configurable Parameters:**
- **Proportion Thresholds:** How dominant a treatment must be to assign a label (default: 0.55 for all classes)
  - `Pos_p`: Threshold for "Good Law" label
  - `Neg_p`: Threshold for "Bad Law" label
  - `Neu_p`: Threshold for "Moderate Law" label
  - `Unk_p`: Threshold for "Unknown" label
- **Time Weighting:** How much recent citations matter (default: MAX_WEIGHT = 2.5)
  - Recent citations get higher weight using exponential decay
- **Time Window:** Date range for citations (default: earliest case to latest case)
- **Jurisdiction Weights:** Extra weight for specific jurisdictions (e.g., `{"California": 1.5}`)
- **Court Strategy:**
  - `lower_level_court=True`: "Walk down" strategy (if highest court is mixed, check lower courts)
  - `lower_level_court=False`: Only use highest court level
- **Label Priority:** Tie-breaking order (default: Unknown > Negative > Neutral > Positive)

**Example Configuration:**
```python
from case_labeler import label_all_cases

label_all_cases(
    force=True,
    echo=True,
    label_thresholds={"Pos_p": 0.60, "Neg_p": 0.60, "Neu_p": 0.55, "Unk_p": 0.55},
    time_weight=[1.0, 3.0],  # MAX_WEIGHT = 3.0 for recent citations
    jurisdictions={"California": 1.5, "New York": 1.2},
    lower_level_court=True,
    include_unknown=True
)
```

---

## Notes: Limitations and Next Steps

**Current Limitations:**
- **Scope:** Limited to ADA cases only (3,500 cases); would need broader dataset for general legal research
- **Citation Coverage:** Snippet extraction finds ~70-80% of citations (some "Id." references and complex citation formats missed)
- **Classification Accuracy:** 67% accuracy leaves room for improvement; ensemble voting helps but individual model errors compound
- **Computational Cost:** LLM ensemble classification is slow (~3 models × 5,500 edges); not optimized for real-time updates
- **Court Hierarchy:** Assumes linear court levels (1-5); doesn't fully capture nuanced jurisdictional relationships (e.g., circuit splits)
- **Recency Bias:** Time-weighting favors recent citations but doesn't distinguish landmark cases from routine applications

**Next Steps:**
- **Expand Dataset:** Add more jurisdictions (contract law, tort law, criminal law) to test generalizability
- **Improve Classification:** Fine-tune smaller open-source models (Llama 3.1, Mistral) on legal citation data to reduce cost and increase accuracy
- **Real-Time Updates:** Integrate CourtListener webhooks to automatically fetch new cases and update citations
- **Citation Graph Analysis:** Add PageRank or citation network metrics to identify influential cases beyond treatment counts
- **User Feedback Loop:** Allow attorneys to correct mislabeled citations and retrain models with human-in-the-loop
- **Deployment:** Containerize with Docker, deploy on AWS (ECS/EKS) with Neo4j cluster for production scalability

---

## Credits / Data / Licenses

**Data Sources:**
- **ADAH (Americans with Disabilities Act Handbook):** PDF of ADA case names used as seed cases
- **CourtListener API:** Free legal case database ([free.law](https://free.law/projects/courtlistener/)) for case metadata, opinions, and citations
  - Usage: Academic/research purposes under CourtListener Terms of Service

**LLM Providers:**
- **Amazon Bedrock:** Claude 3.5 Sonnet, Mistral-7B, Llama 3-70B, Titan Embeddings
  - Used under AWS account with pay-per-use pricing

**Frameworks and Tools:**
- **LangChain / LangGraph:** MIT License
- **Streamlit:** Apache License 2.0
- **Neo4j Community Edition:** GPLv3 License (or Neo4j Aura cloud service)

**Client Partner:**
- **Wolters Kluwer:** Global leader in professional information services for legal, tax, healthcare, and compliance markets
- Project conducted as UC Berkeley School of Information capstone (August 2025 - December 2025)

**License:**
- This project is released under the **MIT License** for portfolio and educational purposes

---

## Team Members

| Name | Email | LinkedIn |
|------|--------|----------|
| **Bryan Guan** | bryguan@berkeley.edu | [LinkedIn](https://www.linkedin.com/in/bryguan/) |
| **Hunter Tonn** | hunter.tonn@berkeley.edu | [LinkedIn](https://www.linkedin.com/in/hunter-tonn/) |
| **Kent Bourgoing** | kent1bp@berkeley.edu | [LinkedIn](https://www.linkedin.com/in/kent-bourgoing-124444168/) |
| **Simran Gill** | simran.gill@berkeley.edu | [LinkedIn](https://www.linkedin.com/in/simran-k-gill/) |
| **Xueying (Wendy) Tian** | xtian9@berkeley.edu | [LinkedIn](https://www.linkedin.com/in/wendy) |

---

## Additional Documentation

For detailed technical documentation on each pipeline step, see:
- **Knowledge Graph Construction:** `Knowledge Graph/JSON_to_CSV_Converter.ipynb` (includes Neo4j schema and import instructions)
- **Citation Classification:** `Citation Classifier/evaluation_pipeline.ipynb` (includes prompt templates and evaluation metrics)
- **Case Labeling Algorithm:** `Case Classifier/case_labeler.py` (includes mathematical formulas and configuration examples)
- **GraphRAG Architecture:** `UI/GraphRAG_and_Agents.ipynb` (includes agent tool definitions and conversation flows)
