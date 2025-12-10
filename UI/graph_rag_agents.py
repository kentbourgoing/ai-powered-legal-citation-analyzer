"""
GraphRAG retrieval functions and LangGraph agents for the ADAH legal UI.

- Connects to Neo4j and AWS Bedrock (Claude + Titan).
- Exposes reusable GraphRAG pipelines:
  - Opinion text semantic search
  - Scenario → ranked cases (Good / Bad / Moderate law)
  - Topic → list of cases (via vector + graph metadata)
  - Text-to-Cypher database querying
  - Case-level citation tools (incoming / outgoing, sentiment filters)
  - Topic + court / jurisdiction filters
- Builds agents that can be called from the UI.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from dotenv import load_dotenv
from neo4j import GraphDatabase

from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector, GraphCypherQAChain
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder


# ------------------------------------------------------------------------------
# 1. Environment and connections
# ------------------------------------------------------------------------------

# Adjust path if your .env is elsewhere.
load_dotenv("../.env", override=True)

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# AWS / Bedrock
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
CLAUDE_MODEL_ID = os.getenv(
    "CLAUDE_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
)
TITAN_MODEL_ID = os.getenv(
    "TITAN_MODEL_ID", "amazon.titan-embed-text-v2:0"
)

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise RuntimeError("Missing Neo4j environment variables in .env")

# Low-level Neo4j driver (optional, mostly for debugging)
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
)
driver.verify_connectivity()

# LangChain Neo4j wrapper
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

# Claude as LLM (chat)
llm = ChatBedrock(
    model_id=CLAUDE_MODEL_ID,
    region_name=AWS_REGION,
)

# Titan as embedder
embedding_model = BedrockEmbeddings(
    model_id=TITAN_MODEL_ID,
    region_name=AWS_REGION,
)

# ------------------------------------------------------------------------------
# 2. Shared agent system prompt and ChatPromptTemplate
# ------------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """You are a legal research assistant.

You can use tools to look up information, but you must NOT mention tools,
tool names, function names, or that you are calling a tool in your replies.

When you answer:
- Focus on U.S. ADA / disability law context based on the data you are given.
- Give a clear, concise answer to the user’s question.
- Do not describe internal steps, tools, or API calls.
"""

react_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_SYSTEM_PROMPT),
        # LangGraph passes the running conversation as `messages`
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ------------------------------------------------------------------------------
# 3. Vector stores (GraphRAG building blocks)
# ------------------------------------------------------------------------------

# We assume:
# - Label: OpinionChunk
# - Text property: text
# - Embedding property: embedding
# - Vector index: chunkEmbeddings

# 3.1 Simple vector store: chunk-level semantic search only
opinion_vector = Neo4jVector.from_existing_index(
    embedding_model,
    graph=graph,
    index_name="chunkEmbeddings",
    text_node_property="text",
    embedding_node_property="embedding",
)

# 3.2 Vector store with OpinionChunk → Case linkage (internal, for scenario ranking)
scenario_retrieval_query = """
RETURN
  node.text AS text,
  score,
  {
    node_element_id: elementId(node),
    case_id: node.case_id,
    opinion_type: node.opinion_type,
    opinion_author: node.opinion_author
  } AS metadata
ORDER BY score DESC
"""

opinion_vector_context = Neo4jVector.from_existing_index(
    embedding_model,
    graph=graph,
    index_name="chunkEmbeddings",
    text_node_property="text",
    embedding_node_property="embedding",
    retrieval_query=scenario_retrieval_query,
)

# 3.3 Vector + graph-aware store returning Case-level metadata
#     Used to answer questions like "What cases talk about X topic?"
case_metadata_retrieval_query = """
MATCH (c:Case)-[:HAS_OPINION_CHUNK]->(node)
RETURN
  node.text AS text,
  score,
  {
    case_name: c.name,
    decision_date: c.decision_date,
    court_name: c.court_name,
    jurisdiction_name: c.jurisdiction_name,
    citation_pipe: c.citation_pipe,
    docket_number: c.docket_number,
    court_listener_url: c.court_listener_url,
    opinion_summary: c.opinion_summary
  } AS metadata
ORDER BY score DESC
"""

opinion_vector_case_metadata = Neo4jVector.from_existing_index(
    embedding_model,
    graph=graph,
    index_name="chunkEmbeddings",
    text_node_property="text",
    embedding_node_property="embedding",
    retrieval_query=case_metadata_retrieval_query,
)

# ------------------------------------------------------------------------------
# 4. Text-to-Cypher (GraphCypherQAChain)
# ------------------------------------------------------------------------------

cypher_template = """Task: Generate a Cypher statement to query a Neo4j database.

Instructions:
- Use only labels, relationship types, and properties present in the schema.
- Do not invent labels or properties.
- Return ONLY a Cypher query, with no explanations or extra text.

The user's question is:
{question}
"""

cypher_prompt = PromptTemplate(
    input_variables=["question"],
    template=cypher_template,
)

cypher_qa = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=llm,
    cypher_llm=llm,
    cypher_prompt=cypher_prompt,
    return_direct=True,
    verbose=False,
    # IMPORTANT: this must be True or the chain raises at __init__
    allow_dangerous_requests=True,
)

# ------------------------------------------------------------------------------
# 5. GraphRAG helper functions (UI-friendly primitives)
# ------------------------------------------------------------------------------


def semantic_search_opinion_chunks(question: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Plain semantic search over OpinionChunk.text.

    Returns a list of dicts with text and metadata (metadata is minimal,
    coming from Neo4jVector default behavior).
    """
    docs = opinion_vector.similarity_search(question, k=k)
    return [
        {"text": d.page_content, "metadata": d.metadata}
        for d in docs
    ]


def run_cypher_qa(question: str) -> Any:
    """
    Run a Text-to-Cypher query using GraphCypherQAChain.
    Returns whatever the chain returns (usually a dict/primitive).
    """
    return cypher_qa.invoke({"query": question})


def _recommend_cases_for_scenario_with_label(
    scenario: str,
    case_label: str,
    k_chunks: int = 40,
    max_cases: int = 5,
    include_snippets: bool = True,
) -> Dict[str, Any]:
    """
    Internal helper: scenario → cases, filtered by a specific case_label.

    Steps:
    1. Semantic search over OpinionChunk (vector + graph context).
    2. Aggregate candidate chunks by case_id.
    3. Filter cases to the given case_label.
    4. Rank by court level (higher first), recency, then incoming citations.
    5. Return top `max_cases` with metadata and representative snippets.
    """
    # Step 1: retrieve candidate chunks
    docs = opinion_vector_context.similarity_search(scenario, k=k_chunks)

    case_to_snippets: Dict[Any, List[str]] = defaultdict(list)
    case_ids: set[Any] = set()

    for d in docs:
        meta = d.metadata or {}
        case_id = meta.get("case_id")
        if case_id is None:
            continue
        case_ids.add(case_id)
        if include_snippets and len(case_to_snippets[case_id]) < 3:
            case_to_snippets[case_id].append(d.page_content)

    if not case_ids:
        return {"scenario": scenario, "case_label": case_label, "cases": []}

    # Step 2–4: filter by case_label and rank cases in Cypher
    rank_query = """
    MATCH (c:Case)
    WHERE c.id IN $case_ids
      AND c.case_label = $case_label
    OPTIONAL MATCH (c)-[:HEARD_IN]->(court:Court)
    OPTIONAL MATCH (:Case)-[cit:CITES_TO]->(c)
    WITH c, court,
         count(cit) AS incoming_citations,
         sum(CASE WHEN cit.treatment_label = 'Positive' THEN 1 ELSE 0 END) AS positive_citations,
         sum(CASE WHEN cit.treatment_label = 'Neutral' THEN 1 ELSE 0 END) AS neutral_citations,
         sum(CASE WHEN cit.treatment_label = 'Negative' THEN 1 ELSE 0 END) AS negative_citations,
         sum(CASE WHEN cit.treatment_label = 'Unknown' THEN 1 ELSE 0 END) AS unknown_citations
    WITH
      c,
      incoming_citations,
      positive_citations,
      neutral_citations,
      negative_citations,
      unknown_citations,
      coalesce(court.court_level, 0) AS court_level_num,
      court
    RETURN
      c.id AS case_id,
      c.name AS case_name,
      c.decision_date AS decision_date,
      c.court_name AS court_name,
      c.jurisdiction_name AS jurisdiction_name,
      court_level_num AS court_level,
      incoming_citations AS incoming_citations,
      positive_citations AS positive_citations,
      neutral_citations AS neutral_citations,
      negative_citations AS negative_citations,
      unknown_citations AS unknown_citations,
      c.citation_pipe AS citation_pipe,
      c.docket_number AS docket_number,
      c.court_listener_url AS court_listener_url,
      c.opinion_summary AS opinion_summary
    ORDER BY
      court_level_num DESC,
      decision_date DESC,
      incoming_citations DESC
    LIMIT $limit
    """

    rows = graph.query(
        rank_query,
        params={
            "case_ids": list(case_ids),
            "limit": max_cases,
            "case_label": case_label,
        },
    )

    cases: List[Dict[str, Any]] = []
    for row in rows:
        cid = row.get("case_id")
        # Do NOT expose internal case_id to the UI
        case_entry = {
            "case_name": row.get("case_name"),
            "decision_date": row.get("decision_date"),
            "court_name": row.get("court_name"),
            "jurisdiction_name": row.get("jurisdiction_name"),
            "court_level": row.get("court_level"),
            "incoming_citations": row.get("incoming_citations"),
            "positive_citations": row.get("positive_citations"),
            "neutral_citations": row.get("neutral_citations"),
            "negative_citations": row.get("negative_citations"),
            "unknown_citations": row.get("unknown_citations"),
            "citation_pipe": row.get("citation_pipe"),
            "docket_number": row.get("docket_number"),
            "court_listener_url": row.get("court_listener_url"),
            "opinion_summary": row.get("opinion_summary"),
            "snippets": case_to_snippets.get(cid, []),
        }
        cases.append(case_entry)

    return {
        "scenario": scenario,
        "case_label": case_label,
        "cases": cases,
    }


def recommend_cases_for_scenario(
    scenario: str,
    k_chunks: int = 40,
    max_cases: int = 5,
    include_snippets: bool = True,
) -> Dict[str, Any]:
    """
    Scenario → top "Good law" cases.

    Uses vector search over OpinionChunk + graph features
    (court level, recency, incoming citations and their treatment labels).
    """
    return _recommend_cases_for_scenario_with_label(
        scenario=scenario,
        case_label="Good",
        k_chunks=k_chunks,
        max_cases=max_cases,
        include_snippets=include_snippets,
    )


def recommend_bad_cases_for_scenario(
    scenario: str,
    k_chunks: int = 40,
    max_cases: int = 5,
    include_snippets: bool = True,
) -> Dict[str, Any]:
    """
    Scenario → top "Bad law" cases (case_label = 'Bad').
    """
    return _recommend_cases_for_scenario_with_label(
        scenario=scenario,
        case_label="Bad",
        k_chunks=k_chunks,
        max_cases=max_cases,
        include_snippets=include_snippets,
    )


def recommend_moderate_cases_for_scenario(
    scenario: str,
    k_chunks: int = 40,
    max_cases: int = 5,
    include_snippets: bool = True,
) -> Dict[str, Any]:
    """
    Scenario → top "Moderate law" cases (case_label = 'Moderate').
    """
    return _recommend_cases_for_scenario_with_label(
        scenario=scenario,
        case_label="Moderate",
        k_chunks=k_chunks,
        max_cases=max_cases,
        include_snippets=include_snippets,
    )


def find_cases_for_topic(
    topic: str,
    k_chunks: int = 40,
    max_cases: int = 10,
    max_snippets_per_case: int = 3,
) -> Dict[str, Any]:
    """
    Given a free-text topic, return a deduplicated list of cases that discuss it.

    Uses:
    - Vector search over OpinionChunk.text
    - Joins back to Case metadata via (:Case)-[:HAS_OPINION_CHUNK]->(:OpinionChunk)
    """
    docs = opinion_vector_case_metadata.similarity_search(topic, k=k_chunks)

    cases_by_key: Dict[Tuple[str, Any], Dict[str, Any]] = {}

    for rank, d in enumerate(docs):
        meta = d.metadata or {}
        case_name = meta.get("case_name")
        citation_pipe = meta.get("citation_pipe")
        key = (case_name, citation_pipe)

        if key not in cases_by_key:
            cases_by_key[key] = {
                "case_name": case_name,
                "decision_date": meta.get("decision_date"),
                "court_name": meta.get("court_name"),
                "jurisdiction_name": meta.get("jurisdiction_name"),
                "citation_pipe": citation_pipe,
                "docket_number": meta.get("docket_number"),
                "court_listener_url": meta.get("court_listener_url"),
                "opinion_summary": meta.get("opinion_summary"),
                "snippets": [d.page_content],
                "_best_rank": rank,
            }
        else:
            entry = cases_by_key[key]
            if len(entry["snippets"]) < max_snippets_per_case:
                entry["snippets"].append(d.page_content)
            entry["_best_rank"] = min(entry["_best_rank"], rank)

    if not cases_by_key:
        return {"topic": topic, "cases": []}

    # Sort by best_rank (lower rank = more similar)
    cases = sorted(cases_by_key.values(), key=lambda c: c["_best_rank"])

    # Drop internal field and limit number of cases
    for c in cases:
        c.pop("_best_rank", None)

    return {"topic": topic, "cases": cases[:max_cases]}


# ------------------------------------------------------------------------------
# 5.b Case resolution + citation helpers
# ------------------------------------------------------------------------------


def _resolve_case_by_name(case_name: str) -> Optional[Dict[str, Any]]:
    """
    Resolve a case by (partial) name, preferring higher courts and more recent decisions.

    Returns a dict with basic case metadata or None if nothing matches.
    """
    query = """
    MATCH (c:Case)
    OPTIONAL MATCH (c)-[:HEARD_IN]->(ct:Court)
    WHERE toLower(c.name) CONTAINS toLower($name)
    RETURN
      c.id AS case_id,
      c.name AS case_name,
      c.decision_date AS decision_date,
      c.citation_pipe AS citation_pipe,
      c.court_name AS court_name,
      c.jurisdiction_name AS jurisdiction_name,
      coalesce(ct.court_level, 0) AS court_level
    ORDER BY court_level DESC, decision_date DESC
    LIMIT 1
    """
    rows = graph.query(query, params={"name": case_name})
    return rows[0] if rows else None


def _get_citations_for_case(
    case_name: str,
    direction: str,
    treatment_label: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Shared helper for tools 1–8.

    direction: "incoming" (cases that cite A) or "outgoing" (cases that A cites).
    treatment_label: None for all, or "Positive" / "Negative" / "Neutral".
    """
    resolved = _resolve_case_by_name(case_name)
    if not resolved:
        return {
            "case_name_query": case_name,
            "direction": direction,
            "treatment_label": treatment_label,
            "citations": [],
            "message": f"No case found matching name '{case_name}' in the database.",
        }

    case_id = resolved.get("case_id")
    sentiment = treatment_label or ""

    if direction == "incoming":
        query = """
        MATCH (target:Case {id: $case_id})
        MATCH (citing:Case)-[r:CITES_TO]->(target)
        OPTIONAL MATCH (citing)-[:HEARD_IN]->(cCourt:Court)
        WHERE ($sentiment = '' OR r.treatment_label = $sentiment)
        RETURN
          citing.name AS other_case_name,
          citing.decision_date AS decision_date,
          citing.citation_pipe AS citation_pipe,
          citing.court_name AS court_name,
          citing.jurisdiction_name AS jurisdiction_name,
          citing.docket_number AS docket_number,
          citing.court_listener_url AS court_listener_url,
          r.treatment_label AS treatment_label,
          r.treatment_rationale AS treatment_rationale,
          coalesce(cCourt.court_level, 0) AS court_level
        ORDER BY court_level DESC, decision_date DESC
        LIMIT $limit
        """
    else:  # "outgoing"
        query = """
        MATCH (source:Case {id: $case_id})
        MATCH (source)-[r:CITES_TO]->(cited:Case)
        OPTIONAL MATCH (cited)-[:HEARD_IN]->(cCourt:Court)
        WHERE ($sentiment = '' OR r.treatment_label = $sentiment)
        RETURN
          cited.name AS other_case_name,
          cited.decision_date AS decision_date,
          cited.citation_pipe AS citation_pipe,
          cited.court_name AS court_name,
          cited.jurisdiction_name AS jurisdiction_name,
          cited.docket_number AS docket_number,
          cited.court_listener_url AS court_listener_url,
          r.treatment_label AS treatment_label,
          r.treatment_rationale AS treatment_rationale,
          coalesce(cCourt.court_level, 0) AS court_level
        ORDER BY court_level DESC, decision_date DESC
        LIMIT $limit
        """

    rows = graph.query(
        query,
        params={
            "case_id": case_id,
            "sentiment": sentiment,
            "limit": limit,
        },
    )

    citations: List[Dict[str, Any]] = []
    for row in rows:
        citations.append(
            {
                "case_name": row.get("other_case_name"),
                "decision_date": row.get("decision_date"),
                "court_name": row.get("court_name"),
                "jurisdiction_name": row.get("jurisdiction_name"),
                "court_level": row.get("court_level"),
                "citation_pipe": row.get("citation_pipe"),
                "docket_number": row.get("docket_number"),
                "treatment_label": row.get("treatment_label"),
                "treatment_rationale": row.get("treatment_rationale"),
                "court_listener_url": row.get("court_listener_url"),
            }
        )

    base_dir = "incoming" if direction == "incoming" else "outgoing"
    resolved_name = resolved.get("case_name")

    if not citations:
        if treatment_label:
            msg = (
                f"Case '{resolved_name}' does not have "
                f"{treatment_label.lower()} {base_dir} citations in the database."
            )
        else:
            msg = f"Case '{resolved_name}' does not have any {base_dir} citations in the database."
    else:
        if treatment_label:
            msg = (
                f"Top {len(citations)} {treatment_label.lower()} {base_dir} citations "
                f"for case '{resolved_name}', ordered by court level and decision date."
            )
        else:
            msg = (
                f"Top {len(citations)} {base_dir} citations for case '{resolved_name}', "
                "ordered by court level and decision date."
            )

    return {
        "case_name_query": case_name,
        "resolved_case": {
            "case_name": resolved.get("case_name"),
            "citation_pipe": resolved.get("citation_pipe"),
            "court_name": resolved.get("court_name"),
            "jurisdiction_name": resolved.get("jurisdiction_name"),
            "decision_date": resolved.get("decision_date"),
            "court_level": resolved.get("court_level"),
        },
        "direction": base_dir,
        "treatment_label": treatment_label,
        "message": msg,
        "citations": citations,
    }


# ------------------------------------------------------------------------------
# 5.c Topic filters by court / jurisdiction
# ------------------------------------------------------------------------------


def find_cases_for_topic_from_court(
    topic: str,
    court_name: str,
    k_chunks: int = 60,
    max_cases: int = 10,
    max_snippets_per_case: int = 3,
) -> Dict[str, Any]:
    """
    Topic → cases whose opinions discuss it, filtered to a given court_name
    (case-insensitive substring match).
    """
    docs = opinion_vector_case_metadata.similarity_search(topic, k=k_chunks)
    court_name_l = court_name.lower()

    cases_by_key: Dict[Tuple[str, Any], Dict[str, Any]] = {}

    for rank, d in enumerate(docs):
        meta = d.metadata or {}
        c_name = meta.get("court_name") or ""
        if court_name_l not in c_name.lower():
            continue

        case_name = meta.get("case_name")
        citation_pipe = meta.get("citation_pipe")
        key = (case_name, citation_pipe)

        if key not in cases_by_key:
            cases_by_key[key] = {
                "case_name": case_name,
                "decision_date": meta.get("decision_date"),
                "court_name": meta.get("court_name"),
                "jurisdiction_name": meta.get("jurisdiction_name"),
                "citation_pipe": citation_pipe,
                "docket_number": meta.get("docket_number"),
                "court_listener_url": meta.get("court_listener_url"),
                "opinion_summary": meta.get("opinion_summary"),
                "snippets": [d.page_content],
                "_best_rank": rank,
            }
        else:
            entry = cases_by_key[key]
            if len(entry["snippets"]) < max_snippets_per_case:
                entry["snippets"].append(d.page_content)
            entry["_best_rank"] = min(entry["_best_rank"], rank)

    if not cases_by_key:
        return {
            "topic": topic,
            "court_name": court_name,
            "cases": [],
            "message": f"No cases found from court '{court_name}' that matched the topic.",
        }

    cases = sorted(cases_by_key.values(), key=lambda c: c["_best_rank"])
    for c in cases:
        c.pop("_best_rank", None)

    return {
        "topic": topic,
        "court_name": court_name,
        "cases": cases[:max_cases],
    }


def find_cases_for_topic_under_jurisdiction(
    topic: str,
    jurisdiction_name: str,
    k_chunks: int = 60,
    max_cases: int = 10,
    max_snippets_per_case: int = 3,
) -> Dict[str, Any]:
    """
    Topic → cases whose opinions discuss it, filtered to a given jurisdiction_name
    (case-insensitive substring match).
    """
    docs = opinion_vector_case_metadata.similarity_search(topic, k=k_chunks)
    juris_l = jurisdiction_name.lower()

    cases_by_key: Dict[Tuple[str, Any], Dict[str, Any]] = {}

    for rank, d in enumerate(docs):
        meta = d.metadata or {}
        j_name = meta.get("jurisdiction_name") or ""
        if juris_l not in j_name.lower():
            continue

        case_name = meta.get("case_name")
        citation_pipe = meta.get("citation_pipe")
        key = (case_name, citation_pipe)

        if key not in cases_by_key:
            cases_by_key[key] = {
                "case_name": case_name,
                "decision_date": meta.get("decision_date"),
                "court_name": meta.get("court_name"),
                "jurisdiction_name": meta.get("jurisdiction_name"),
                "citation_pipe": citation_pipe,
                "docket_number": meta.get("docket_number"),
                "court_listener_url": meta.get("court_listener_url"),
                "opinion_summary": meta.get("opinion_summary"),
                "snippets": [d.page_content],
                "_best_rank": rank,
            }
        else:
            entry = cases_by_key[key]
            if len(entry["snippets"]) < max_snippets_per_case:
                entry["snippets"].append(d.page_content)
            entry["_best_rank"] = min(entry["_best_rank"], rank)

    if not cases_by_key:
        return {
            "topic": topic,
            "jurisdiction_name": jurisdiction_name,
            "cases": [],
            "message": f"No cases found under jurisdiction '{jurisdiction_name}' that matched the topic.",
        }

    cases = sorted(cases_by_key.values(), key=lambda c: c["_best_rank"])
    for c in cases:
        c.pop("_best_rank", None)

    return {
        "topic": topic,
        "jurisdiction_name": jurisdiction_name,
        "cases": cases[:max_cases],
    }


# ------------------------------------------------------------------------------
# 6. Tools for agents (wrapping GraphRAG functions)
# ------------------------------------------------------------------------------


@tool("Get-graph-database-schema")
def get_schema() -> str:
    """Return the raw schema of the Neo4j graph database."""
    # Neo4jGraph.schema returns a human-readable string
    return graph.schema


@tool("Search-opinion-text")
def search_opinion_text_tool(query: str) -> List[str]:
    """
    Find relevant opinion text snippets for a free-text legal question.

    Uses the simple vector store over OpinionChunk.text and returns only
    raw text snippets (no node metadata).
    """
    docs = opinion_vector.similarity_search(query, k=5)
    return [d.page_content for d in docs]


@tool("Query-database")
def query_database(question: str) -> Any:
    """
    Answer specific factual questions by generating and running Cypher
    over the Neo4j graph.
    """
    return run_cypher_qa(question)


@tool("Recommend-cases-for-scenario")
def recommend_cases_for_scenario_tool(scenario: str) -> Dict[str, Any]:
    """
    Given a legal scenario or argument, return top recommended "Good law" cases
    (case_label = 'Good'), ranked by court level, recency, and incoming citations,
    including counts by treatment_label and representative snippets.
    """
    return recommend_cases_for_scenario(scenario)


@tool("Recommend-bad-cases-for-scenario")
def recommend_bad_cases_for_scenario_tool(scenario: str) -> Dict[str, Any]:
    """
    Given a legal scenario or argument, return top recommended "Bad law" cases
    (case_label = 'Bad'), ranked by court level, recency, and incoming citations.
    """
    return recommend_bad_cases_for_scenario(scenario)


@tool("Recommend-moderate-cases-for-scenario")
def recommend_moderate_cases_for_scenario_tool(scenario: str) -> Dict[str, Any]:
    """
    Given a legal scenario or argument, return top recommended "Moderate law" cases
    (case_label = 'Moderate'), ranked by court level, recency, and incoming citations.
    """
    return recommend_moderate_cases_for_scenario(scenario)


@tool("List-cases-for-topic")
def list_cases_for_topic_tool(topic: str) -> Dict[str, Any]:
    """
    Given a topic (e.g., "reasonable accommodation"), list cases that discuss it,
    using vector search over opinion text and returning case-level metadata
    plus representative snippets.
    """
    return find_cases_for_topic(topic)



@tool("List-incoming-citations-for-case")
def list_incoming_citations_for_case_tool(case_name: str) -> Dict[str, Any]:
    """
    Given the name of a case, list up to 5 incoming citations
    (cases that cite it), ordered by highest court level and then
    by most recent decision date, for all treatment labels.
    """
    return _get_citations_for_case(
        case_name=case_name,
        direction="incoming",
        treatment_label=None,
        limit=5,
    )


@tool("List-incoming-positive-citations-for-case")
def list_incoming_positive_citations_for_case_tool(case_name: str) -> Dict[str, Any]:
    """
    Given the name of a case, list up to 5 incoming citations where
    treatment_label = 'Positive', ordered by highest court level and then
    by most recent decision date.
    """
    return _get_citations_for_case(
        case_name=case_name,
        direction="incoming",
        treatment_label="Positive",
        limit=5,
    )


@tool("List-incoming-negative-citations-for-case")
def list_incoming_negative_citations_for_case_tool(case_name: str) -> Dict[str, Any]:
    """
    Given the name of a case, list up to 5 incoming citations where
    treatment_label = 'Negative', ordered by highest court level and then
    by most recent decision date.
    """
    return _get_citations_for_case(
        case_name=case_name,
        direction="incoming",
        treatment_label="Negative",
        limit=5,
    )


@tool("List-incoming-neutral-citations-for-case")
def list_incoming_neutral_citations_for_case_tool(case_name: str) -> Dict[str, Any]:
    """
    Given the name of a case, list up to 5 incoming citations where
    treatment_label = 'Neutral', ordered by highest court level and then
    by most recent decision date.
    """
    return _get_citations_for_case(
        case_name=case_name,
        direction="incoming",
        treatment_label="Neutral",
        limit=5,
    )



@tool("List-outgoing-citations-for-case")
def list_outgoing_citations_for_case_tool(case_name: str) -> Dict[str, Any]:
    """
    Given the name of a case, list up to 5 cases that it cites
    (outgoing :CITES_TO relationships), ordered by highest court level and then
    by most recent decision date, for all treatment labels.
    """
    return _get_citations_for_case(
        case_name=case_name,
        direction="outgoing",
        treatment_label=None,
        limit=5,
    )


@tool("List-outgoing-positive-citations-for-case")
def list_outgoing_positive_citations_for_case_tool(case_name: str) -> Dict[str, Any]:
    """
    Given the name of a case, list up to 5 cases that it cites positively
    (treatment_label = 'Positive'), ordered by highest court level and then
    by most recent decision date.
    """
    return _get_citations_for_case(
        case_name=case_name,
        direction="outgoing",
        treatment_label="Positive",
        limit=5,
    )


@tool("List-outgoing-negative-citations-for-case")
def list_outgoing_negative_citations_for_case_tool(case_name: str) -> Dict[str, Any]:
    """
    Given the name of a case, list up to 5 cases that it cites negatively
    (treatment_label = 'Negative'), ordered by highest court level and then
    by most recent decision date.
    """
    return _get_citations_for_case(
        case_name=case_name,
        direction="outgoing",
        treatment_label="Negative",
        limit=5,
    )


@tool("List-outgoing-neutral-citations-for-case")
def list_outgoing_neutral_citations_for_case_tool(case_name: str) -> Dict[str, Any]:
    """
    Given the name of a case, list up to 5 cases that it cites neutrally
    (treatment_label = 'Neutral'), ordered by highest court level and then
    by most recent decision date.
    """
    return _get_citations_for_case(
        case_name=case_name,
        direction="outgoing",
        treatment_label="Neutral",
        limit=5,
    )


@tool("List-cases-for-topic-from-court")
def list_cases_for_topic_from_court_tool(
    topic: str, court_name: str
) -> Dict[str, Any]:
    """
    Given a topic and a court name, return cases that discuss the topic
    and are from that court (case-insensitive match on court_name),
    including case metadata and representative snippets.
    """
    return find_cases_for_topic_from_court(topic, court_name)


@tool("List-cases-for-topic-under-jurisdiction")
def list_cases_for_topic_under_jurisdiction_tool(
    topic: str, jurisdiction_name: str
) -> Dict[str, Any]:
    """
    Given a topic and a jurisdiction name, return cases that discuss the topic
    and fall under that jurisdiction (case-insensitive match on jurisdiction_name),
    including case metadata and representative snippets.
    """
    return find_cases_for_topic_under_jurisdiction(topic, jurisdiction_name)


# ------------------------------------------------------------------------------
# 7. Agents (ready to be used in the UI)
# ------------------------------------------------------------------------------

# Schema-only agent
schema_tools = [get_schema]
schema_agent = create_react_agent(
    llm,
    schema_tools,
    prompt=react_prompt,
)

# Scenario / topic-focused agent (optional)
scenario_tools = [
    recommend_cases_for_scenario_tool,
    recommend_bad_cases_for_scenario_tool,
    recommend_moderate_cases_for_scenario_tool,
    list_cases_for_topic_tool,
    list_cases_for_topic_from_court_tool,
    list_cases_for_topic_under_jurisdiction_tool,
    query_database,
]
scenario_agent = create_react_agent(
    llm,
    scenario_tools,
    prompt=react_prompt,
)

# Multi-tool agent with all capabilities for the UI
multi_tools = [
    get_schema,
    search_opinion_text_tool,
    query_database,
    recommend_cases_for_scenario_tool,
    recommend_bad_cases_for_scenario_tool,
    recommend_moderate_cases_for_scenario_tool,
    list_cases_for_topic_tool,
    list_incoming_citations_for_case_tool,
    list_incoming_positive_citations_for_case_tool,
    list_incoming_negative_citations_for_case_tool,
    list_incoming_neutral_citations_for_case_tool,
    list_outgoing_citations_for_case_tool,
    list_outgoing_positive_citations_for_case_tool,
    list_outgoing_negative_citations_for_case_tool,
    list_outgoing_neutral_citations_for_case_tool,
    list_cases_for_topic_from_court_tool,
    list_cases_for_topic_under_jurisdiction_tool,
]
multi_tool_agent = create_react_agent(
    llm,
    multi_tools,
    prompt=react_prompt,
)

# ------------------------------------------------------------------------------
# 8. Simple helper to call agents from the UI (non-streaming)
# ------------------------------------------------------------------------------


def invoke_agent(agent, user_message: str) -> Dict[str, Any]:
    """
    Call a LangGraph agent with a single user message.

    Returns the full state dict; the latest AI message is under
    result["messages"][-1].
    """
    return agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )


def invoke_agent_text(agent, user_message: str) -> str:
    """
    Convenience helper: return just the text content of the last AI message.
    """
    state = invoke_agent(agent, user_message)
    if not state or "messages" not in state or not state["messages"]:
        return ""

    last_msg = state["messages"][-1]
    content = getattr(last_msg, "content", "")

    if isinstance(content, str):
        return content

    # If Bedrock returns a list of content blocks, join any "text" fields
    try:
        parts = []
        for part in content:
            text = part.get("text")
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else str(content)
    except Exception:
        return str(content)


__all__ = [
    # Core graph objects
    "graph",
    "driver",
    "llm",
    "embedding_model",
    "opinion_vector",
    "opinion_vector_case_metadata",
    # GraphRAG helpers
    "semantic_search_opinion_chunks",
    "run_cypher_qa",
    "recommend_cases_for_scenario",
    "recommend_bad_cases_for_scenario",
    "recommend_moderate_cases_for_scenario",
    "find_cases_for_topic",
    "find_cases_for_topic_from_court",
    "find_cases_for_topic_under_jurisdiction",
    # Agents
    "schema_agent",
    "multi_tool_agent",
    "scenario_agent",
    # Agent helpers
    "invoke_agent",
    "invoke_agent_text",
]
