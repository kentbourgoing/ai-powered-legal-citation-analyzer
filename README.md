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

The system evaluates "Guy Amir v. St. Louis University" as "Good" law based on 46 incoming citations, primarily from Courts of Appeals. The detailed rationale explains how time-weighted citation analysis and court hierarchy inform the final label.

### Configuration Panel
![Configuration Panel](UI%20Images/configuration-panel.png)

The application allows users to configure case labeling parameters, including proportion thresholds, time-based weighting, and court hierarchy strategies to customize analysis for different practice areas.

### Chatbot Interface - Citation Analysis
![Chatbot Interface](UI%20Images/chatbot-interface.png)

Chatbot demonstration: This example shows the system identifying cases that criticize "Access Now, Inc. v. Southwest Airlines Co." and explaining how courts narrowed its holding, illustrating the platform's ability to deliver transparent, paragraph-level legal reasoning.

---

## What we delivered

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
ai-powered-legal-citation-analyzer/
├── Knowledge Graph/ # Step 1: Build Neo4j knowledge graph
│ ├── ADAH_API_Extract.ipynb # Extract case names, fetch CourtListener data
│ ├── JSON_to_CSV_Converter.ipynb # Format for Neo4j import
│ ├── Summarization_Pipeline.ipynb # Generate case summaries (Mistral-7B)
│ └── EDA.ipynb # Exploratory data analysis
│
├── Citation Classifier/ # Step 2: Classify citation edges
│ ├── Snippet_Retriever.ipynb # Extract citation context snippets
│ ├── Edge_Classifier_Snippet_Method_Ensemble.ipynb # Ensemble voting (BEST)
│ ├── Edge_Classifier_Snippet_Method_Claude.ipynb
│ ├── Edge_Classifier_Snippet_Method_Llama3.ipynb
│ ├── Edge_Classifier_Snippet_Method_Mistral.ipynb
│ └── evaluation_pipeline.ipynb # Evaluate classifier performance
│
├── Case Classifier/ # Step 3: Label cases (Good/Bad/Moderate)
│ ├── Case_Labeler.ipynb # Run case labeling workflow
│ └── case_labeler.py # Time-weighted aggregation algorithm
│
├── UI/ # Step 4: Streamlit web app + GraphRAG
│ ├── app.py # Main Streamlit application (ENTRY POINT)
│ ├── graph_rag_agents.py # LangGraph agents + tools
│ ├── embed_opinion_chunks.ipynb # Create Titan embeddings
│ ├── GraphRAG_and_Agents.ipynb # Test GraphRAG agents
│ └── run_app.ipynb # Launch app from notebook
│
├── UI Images/ # Screenshots for README
│ ├── homepage.png # Application homepage screenshot
│ ├── case-detail-view.png # Case detail view screenshot
│ ├── configuration-panel.png # Configuration panel screenshot
│ └── chatbot-interface.png # Chatbot interface screenshot
│
├── Data/ # Data files
├── Slides/ # Project presentation slides
├── requirement.txt # Python dependencies
└── README.md # Project documentation
```

---

## Prerequisites

Platforms to support data/ LLM/ model training 

- **[Neo4j Database](https://neo4j.com/product/)** - local or cloud instance for graph database
- **[AWS Account](https://aws.amazon.com/console/)**
  - Amazon Bedrock - LLM provider
  - Amazon SageMaker Studio 
  - S3 - data storage
- **[CourtListener](https://free.law/projects/courtlistener/)** - legal case provider 
- Required Python packages (install via `pip install -r requirements.txt`)

### Required Environment Variables

Create a `.env` file in the project root with:

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

## How to Run
## Step 1: Knowledge Graph Construction
**Location:** `Knowledge Graph/`

This step extracts legal case data and builds a Neo4j knowledge graph with cases, citations, and necessary attributes.

### 1.1 Extract Case Names from PDF
**Notebook:** `ADAH_API_Extract.ipynb`
- Extracts clean ADA case names from the ADAH (Americans with Disabilities Act Handbook) PDF
- Normalizes case names (handles `v.`, `vs.`, punctuation variations)
- Outputs a list of case names to `ada_case_names_only.txt`

### 1.2 Fetch Cases from CourtListener API
**Notebook:** `ADAH_API_Extract.ipynb`
- Fetches ADAH "seed" cases by name from CourtListener API
- Downloads cases cited by ADAH cases
- Downloads cases that cite ADAH cases
- Handles rate limiting and retries

### 1.3 Format the Cases Ready for Neo4j Aura 
**Notebook:** `JSON_to_CSV_Converter.ipynb`
- Converts JSON case data to CSV files for Neo4j import
- Prepares the following CSV files:
  - `cases.csv` - Case nodes with properties (id, name, citation, decision_date, court_id, jurisdiction_id, etc.)
  - `courts.csv` - Court nodes with court_level (1-5)
  - `jurisdictions.csv` - Jurisdiction nodes
  - `cites_to.csv` - Citation relationships between cases
  - `opinion_chunks.csv` - OpinionChunk nodes (for long opinion text, if using chunked mode)
  - `case_opinion_edges.csv` - Relationships linking cases to opinion chunks
- These CSV files define the Neo4j graph structure:
  - **Nodes:** `Case`, `Court`, `Jurisdiction`, `OpinionChunk`
  - **Relationships:**
    - `(:Case)-[:CITES_TO]->(:Case)` - citation relationships
    - `(:Case)-[:HEARD_IN]->(:Court)` - court relationships
    - `(:Case)-[:UNDER_JURISDICTION]->(:Jurisdiction)` - jurisdiction relationships
    - `(:Case)-[:HAS_OPINION_CHUNK]->(:OpinionChunk)` - opinion text chunks
- **Important:** Upload the CSV files manually through the Neo4j Aura Data Importer to construct the knowledge graph
- **Note:** The `JSON_to_CSV_Converter.ipynb` notebook contains detailed instructions on the import schema and how to build the graph in Neo4j Aura

### 1.4 Generate Case Summaries
**Notebook:** `Summarization_Pipeline.ipynb`
- Uses Amazon Bedrock (Mistral) to generate single-paragraph summaries
- Writes summaries to `(:Case).opinion_summary`
- Handles long opinions by chunking with token-based splitting

### 1.5 Exploratory Data Analysis
**Notebook:** `EDA.ipynb`
- Analyze the graph structure
- Check data quality
- Visualize relationships

**How to Run:**
1. Start with `ADAH_API_Extract.ipynb` to extract case names and fetch data
2. Run `JSON_to_CSV_Converter.ipynb` to generate CSV files
3. Manually upload the CSV files to Neo4j Aura using the Data Importer (follow instructions in the notebook)
4. Run `Summarization_Pipeline.ipynb` to generate case summaries
5. Use `EDA.ipynb` to explore your data
---

## Step 2: Citation Classification

**Location:** `Citation Classifier/`

This step classifies how cases cite each other by analyzing the context around citations in opinion text.

### 2.1 Extract Citation Snippets

**Notebook:** `Snippet_Retriever.ipynb`

- Scans opinion text for citation references
- Extracts snippets (context windows) around each citation
- Writes snippets to `(:Case)-[:CITES_TO]->(:Case)` edges as `snippet_1`, `snippet_2`, etc.
- Handles multiple citation formats and "Id." references

**Key Features:**
- Multiple search methods (exact citation, relaxed citation, case name matching)
- Merges overlapping snippets
- Expands to sentence boundaries
- Tracks which method found each citation

### 2.2 Classify Citation Treatment

**Notebooks:** 
- `Edge_Classifier_Snippet_Method_Claude.ipynb`
- `Edge_Classifier_Snippet_Method_Ensemble.ipynb`
- `Edge_Classifier_Snippet_Method_Llama3.ipynb`
- `Edge_Classifier_Snippet_Method_Mistral.ipynb`

Each notebook uses a different LLM to classify citation treatment:

- **Positive** - Case is cited favorably (followed, approved, extended)
- **Negative** - Case is cited unfavorably (distinguished, criticized, overruled)
- **Neutral** - Case is cited neutrally (mentioned, referenced without judgment)
- **Unknown** - Treatment cannot be determined

**Process:**
1. Load citation snippets from Neo4j edges
2. Send snippets to LLM with classification prompt
3. Parse LLM response to extract treatment label and rationale
4. Write `treatment_label` and `treatment_rationale` back to edges

### 2.3 Evaluate Classifiers

**Notebook:** `evaluation_pipeline.ipynb`

- Compares classifier performance against ground truth labels
- Generates accuracy, precision, recall metrics
- Produces comparison reports

**How to Run:**

1. Run `Snippet_Retriever.ipynb` to extract citation snippets
2. Choose one of the classifier notebooks (Ensemble recommended for best accuracy)
3. Run the classifier to label all citation edges
4. Use `evaluation_pipeline.ipynb` to evaluate performance

---

## Step 3: Case Classification

**Location:** `Case Classifier/`

This step labels cases (Good, Bad, Moderate, Unknown) based on their citation history using time-weighted analysis.

### 3.1 Label Cases from Citation History

**Notebook:** `Case_Labeler.ipynb`  
**Script:** `case_labeler.py`

The case labeler analyzes incoming citations to each case and assigns a label:

- **Good** - Predominantly positive citations (favorable treatment)
- **Bad** - Predominantly negative citations (unfavorable treatment)
- **Moderate** - Mixed or neutral citations
- **Unknown** - Insufficient citation data

**Algorithm:**
1. Collect all incoming citations for each case
2. Group citations by court level (Supreme Court, Appeals, District, etc.)
3. Apply time-based weighting (recent citations weighted more heavily)
4. Apply jurisdiction weights (optional)
5. Compute weighted proportions of Positive/Negative/Neutral/Unknown treatments
6. Apply thresholds to determine label
7. Use "walk down" strategy: if highest court is mixed, check lower courts

**Configuration Options:**
- Proportion thresholds (how dominant a treatment must be)
- Time-based weighting (how much recent citations matter)
- Time window (date range for citations)
- Jurisdiction weights (extra weight for specific jurisdictions)
- Court selection strategy (highest court only vs. walk down)
- Label priority order (tie-breaking)

**How to Run:**

```python
from case_labeler import label_all_cases

# Default configuration
label_all_cases(
    force=True,              # Re-label all cases (even if already labeled)
    echo=True,               # Print progress
    lower_level_court=True,  # Use walk-down strategy
    include_unknown=True     # Include Unknown treatments in calculations
)

# Custom configuration
label_all_cases(
    force=True,
    echo=True,
    label_thresholds={"Pos_p": 0.60, "Neg_p": 0.60, "Neu_p": 0.55, "Unk_p": 0.55},
    time_weight=[1.0, 3.0],  # MAX_WEIGHT = 3.0
    jurisdictions={"California": 1.5, "New York": 1.2},
    lower_level_court=True
)
```

---

## Step 4: Web UI Development

**Location:** `UI/`

This step builds a Streamlit web application with case lookup and chatbot functionality powered by GraphRAG.

### 4.1 Embed Opinion Chunks

**Notebook:** `embed_opinion_chunks.ipynb`

- Embeds all `OpinionChunk` nodes using Amazon Titan embeddings
- Stores embeddings in `(:OpinionChunk).embedding`
- Creates Neo4j vector index `chunkEmbeddings` for semantic search

**Why:** Enables semantic search over opinion text for GraphRAG queries.

### 4.2 Build GraphRAG Agents

**Notebook:** `GraphRAG_and_Agents.ipynb`  
**Script:** `graph_rag_agents.py`

Creates LangGraph agents with tools for:
- Semantic search over opinion text
- Text-to-Cypher database queries
- Scenario-based case recommendations (Good/Bad/Moderate law)
- Topic-based case discovery
- Citation analysis (incoming/outgoing, filtered by treatment)

**Agent Tools:**
- `Search-opinion-text` - Find relevant opinion snippets
- `Query-database` - Run Cypher queries
- `Recommend-cases-for-scenario` - Find "Good law" cases for a scenario
- `Recommend-bad-cases-for-scenario` - Find "Bad law" cases
- `List-cases-for-topic` - Find cases discussing a topic
- `List-incoming-citations-for-case` - Get citations to a case
- And more...

### 4.3 Build Streamlit App

**Script:** `app.py`

The main Streamlit application with two modes:

#### Case Lookup Mode
- Search for cases by name
- View case details (name, citation, date, court, summary)
- See citation evaluation (Good/Bad/Moderate/Unknown label)
- Browse citing cases with treatment labels and rationales
- Export citing cases to CSV

#### Chatbot Mode
- Natural language questions about ADA cases
- Powered by GraphRAG agents
- Answers include relevant cases and citations
- Can handle:
  - Case summaries
  - Citation treatment questions
  - Scenario-based recommendations
  - Topic exploration
  - Comparative analysis

#### Case Label Configuration
- Adjust case labeling parameters
- Update case labels with new configuration
- See methodology explanation

**How to Run:**

1. **Embed opinion chunks:**
   ```bash
   # Run embed_opinion_chunks.ipynb
   ```

2. **Create vector index in Neo4j:**
   ```cypher
   CREATE VECTOR INDEX chunkEmbeddings IF NOT EXISTS
   FOR (c:OpinionChunk)
   ON c.embedding
   OPTIONS {
     indexConfig: {
       `vector.dimensions`: 1024,
       `vector.similarity_function`: 'cosine'
     }
   }
   ```

3. **Start the Streamlit app:**
   ```bash
   cd UI
   streamlit run app.py
   ```

   Or use the notebook: `run_app.ipynb`

---

## Running the Application

### Complete Setup Workflow

1. **Set up environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   - Create `.env` file with Neo4j and AWS credentials
   - See [Configuration](#configuration) section

3. **Build the knowledge graph:**
   - Follow Step 1 instructions
   - This creates the Neo4j database with cases and relationships

4. **Classify citations:**
   - Follow Step 2 instructions
   - This labels citation edges with treatment labels

5. **Label cases:**
   - Follow Step 3 instructions
   - This assigns Good/Bad/Moderate/Unknown labels to cases

6. **Embed opinion chunks:**
   - Run `UI/embed_opinion_chunks.ipynb`
   - Create vector index in Neo4j

7. **Launch the UI:**
   ```bash
   cd UI
   streamlit run app.py
   ```

### Quick Start (Using Existing Data)

If you already have a populated Neo4j database:

1. Ensure embeddings exist (run `embed_opinion_chunks.ipynb` if needed)
2. Ensure vector index exists (create with Cypher if needed)
3. Launch the app:
   ```bash
   cd UI
   streamlit run app.py
   ```

---

## Configuration

### Streamlit Secrets

For production deployment, configure Streamlit secrets (`.streamlit/secrets.toml`):

```toml
[NEO4J]
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "your_password"
DATABASE = "neo4j"
```

Or use environment variables in your deployment platform.

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
- **Streamlit:** Apache License 2.0
- **Neo4j Community Edition:** GPLv3 License (or Neo4j Aura cloud service)

**Client Partner:**
- **Wolters Kluwer:** Global leader in professional information services for legal, tax, healthcare, and compliance markets
- Project conducted as UC Berkeley School of Information capstone (August 2025 - December 2025)


---

## Team Members

| Name | Email | LinkedIn |
|------|--------|----------|
| **Bryan Guan** | bryguan@berkeley.edu | [LinkedIn](https://www.linkedin.com/in/bryguan/) |
| **Hunter Tonn** | hunter.tonn@berkeley.edu | [LinkedIn](https://www.linkedin.com/in/hunter-tonn/) |
| **Kent Bourgoing** | kent1bp@berkeley.edu | [LinkedIn](https://www.linkedin.com/in/kent-bourgoing-124444168/) |
| **Simran Gill** | simran.gill@berkeley.edu | [LinkedIn](https://www.linkedin.com/in/simran-k-gill/) |
| **Xueying (Wendy) Tian** | xtian9@berkeley.edu | [LinkedIn](https://www.linkedin.com/in/wendy) |


