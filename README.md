# AI-powered Interactive Legal Citation Evaluator

A web-based application for exploring and analyzing U.S. court cases under the Americans with Disabilities Act (ADA). This project uses a Neo4j knowledge graph, AI-powered classifiers, and a Streamlit-based UI to help legal researchers understand how cases are cited and treated by later decisions.

## Overview

The Legal Citation Machine is built using a **4-step approach**:

1. **Knowledge Graph Construction** - Extract and build a Neo4j knowledge graph from legal case data
2. **Citation Classification** - Classify how cases cite each other (Positive, Negative, Neutral, Unknown)
3. **Case Classification** - Label cases based on their citation history (Good, Bad, Moderate, Unknown)
4. **Web UI Development** - Build an interactive Streamlit interface with GraphRAG-powered search and chatbot

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Step 1: Knowledge Graph Construction](#step-1-knowledge-graph-construction)
- [Step 2: Citation Classification](#step-2-citation-classification)
- [Step 3: Case Classification](#step-3-case-classification)
- [Step 4: Web UI Development](#step-4-web-ui-development)
- [Running the Application](#running-the-application)
- [Configuration](#configuration)

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

---

## Project Structure

```
capstone git/
├── Knowledge Graph/          # Step 1: Build the knowledge graph
│   ├── ADAH_API_Extract.ipynb
│   ├── JSON_to_CSV_Converter.ipynb
│   ├── Summarization_Pipeline.ipynb
│   └── EDA.ipynb
│
├── Citation Classifier/      # Step 2: Classify citation edges
│   ├── Snippet_Retriever.ipynb
│   ├── Edge_Classifier_Snippet_Method_Claude.ipynb
│   ├── Edge_Classifier_Snippet_Method_Ensemble.ipynb
│   ├── Edge_Classifier_Snippet_Method_Llama3.ipynb
│   ├── Edge_Classifier_Snippet_Method_Mistral.ipynb
│   └── evaluation_pipeline.ipynb
│
├── Case Classifier/           # Step 3: Label cases based on citations
│   ├── Case_Labeler.ipynb
│   └── case_labeler.py
│
└── UI/                        # Step 4: Build the Streamlit web app
    ├── app.py
    ├── graph_rag_agents.py
    ├── embed_opinion_chunks.ipynb
    ├── GraphRAG_and_Agents.ipynb
    └── run_app.ipynb
```

---

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

### 1.3 Build Neo4j Graph

**Notebook:** `JSON_to_CSV_Converter.ipynb`

- Converts JSON case data to structured format
- Creates Neo4j nodes:
  - `Case` nodes with properties (name, citation, decision_date, court, etc.)
  - `Court` nodes
  - `Jurisdiction` nodes
  - `OpinionChunk` nodes (for long opinion text)
- Creates relationships:
  - `(:Case)-[:CITES_TO]->(:Case)` - citation relationships
  - `(:Case)-[:HEARD_IN]->(:Court)` - court relationships
  - `(:Case)-[:UNDER_JURISDICTION]->(:Jurisdiction)` - jurisdiction relationships
  - `(:Case)-[:HAS_OPINION_CHUNK]->(:OpinionChunk)` - opinion text chunks

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
2. Run `JSON_to_CSV_Converter.ipynb` to build the Neo4j graph
3. Run `Summarization_Pipeline.ipynb` to generate case summaries
4. Use `EDA.ipynb` to explore your data

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
2. Choose one of the classifier notebooks (Claude recommended for best accuracy)
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

### Case Labeling Configuration

The case labeler supports extensive configuration. See `Case Classifier/case_labeler.py` for all options:

- **Proportion Thresholds:** Control how dominant a treatment must be (default: 0.55)
- **Time Weighting:** How much recent citations matter (default: MAX_WEIGHT = 2.5)
- **Time Window:** Date range for citations (default: Q1 to latest)
- **Jurisdiction Weights:** Extra weight for specific jurisdictions
- **Court Strategy:** Highest court only vs. walk down to lower courts
- **Label Priority:** Tie-breaking order (default: Unknown > Negative > Neutral > Positive)

---

## Project Summary

- **Data Source:** Cases extracted from ADAH (Americans with Disabilities Act Handbook) and CourtListener
- **Graph Database:** Neo4j 5.x
- **LLM:** Amazon Bedrock (Claude 3.5 Sonnet, Mistral, Llama3, Titan)
- **UI Framework:** Streamlit
- **RAG Framework:** LangChain + LangGraph

---
