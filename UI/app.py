import os
import sys
import json
import html as html_lib
from datetime import date, datetime
from typing import Dict, List, Any, Optional, Tuple
import base64
import time  # <-- for retry sleep

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase, Driver
from streamlit_searchbox import st_searchbox
import streamlit.components.v1 as components


# ---------------------- Import case labeler ----------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_LABELING_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Case Classifier"))
if CASE_LABELING_DIR not in sys.path:
    sys.path.append(CASE_LABELING_DIR)

from case_labeler import label_all_cases, VALID_JURISDICTIONS  # noqa: E402

# GraphRAG + agents (chatbot)
from graph_rag_agents import multi_tool_agent, invoke_agent_text  # noqa: E402

VALID_JURIS_LIST = sorted(list(VALID_JURISDICTIONS))

CITATION_PAGE_SIZE = 10


# ---------------------- Help panel HTML ----------------------

HELP_PANEL_HTML = """
<div class="help-panel">
  <h2>What this app does</h2>
  <p>
    The <strong>Legal Citation Machine</strong> helps you explore how U.S. courts treat cases
    under the <strong>Americans with Disabilities Act (ADA)</strong>.
  </p>
  <p>
    The app is backed by a Neo4j knowledge graph of ADA-relevant cases. For each case, it shows:
  </p>
  <ul>
    <li>A short <strong>summary</strong> of the decision</li>
    <li>How later cases <strong>treat</strong> it (positively, negatively, neutrally, or in a mixed way)</li>
    <li>A <strong>case-level label</strong> such as <em>Good</em>, <em>Bad</em>, <em>Moderate</em>, or <em>Unknown</em> based on its citation history</li>
  </ul>
  <p>You can use the app in two main ways:</p>
  <ul>
    <li><strong>Case Lookup</strong> – look up a specific case and see its citations and evaluation.</li>
    <li><strong>Chatbot</strong> – ask natural-language questions about ADA cases, concepts, and patterns.</li>
  </ul>

  <h3>Case Lookup</h3>
  <p>Use <strong>Case Lookup</strong> when you already have a case in mind and want details about it and its citation history.</p>
  <ol>
    <li>
      <strong>Select “Case Lookup” mode</strong><br/>
      At the top of the app, select <strong>“Case Lookup”</strong>.
    </li>
    <li>
      <strong>Search for a case</strong><br/>
      In the search box, start typing the case name. For example:<br/>
      <code>Access Now, Inc. v. Southwest Airlines Co.</code><br/>
      The search box will show matching cases as you type. Click the desired case from the dropdown.
    </li>
    <li>
      <strong>Read the case details</strong><br/>
      After selecting a case, the app shows:
      <ul>
        <li><strong>Name</strong> – full case name</li>
        <li><strong>Citation</strong> – reporter citation if available</li>
        <li><strong>Decision Date</strong></li>
        <li><strong>Court</strong> and <strong>Jurisdiction</strong></li>
        <li><strong>Docket Number</strong></li>
        <li><strong>Summary</strong> – a short description of the opinion</li>
        <li><strong>URL</strong> – link to the opinion on an external site (for full text)</li>
      </ul>
    </li>
    <li>
      <strong>Review the citation-based evaluation</strong><br/>
      Below the basic details, you will see:
      <ul>
        <li><strong>Citation Evaluation</strong> – a label such as <em>Good</em>, <em>Bad</em>, <em>Moderate</em>, or <em>Unknown</em>.</li>
        <li><strong>Decision Level</strong> – the court level used for the final label.</li>
        <li><strong>Rationale</strong> – a brief explanation of how the label was derived from citation history.</li>
      </ul>
      You can expand <strong>“Methodology”</strong> to see the high-level rules used for this evaluation.
    </li>
    <li>
      <strong>Explore citing cases</strong><br/>
      Under <strong>“Citing Cases”</strong>, you can:
      <ul>
        <li>See a list of <strong>cases that cite your selected case</strong>, each in an expandable card.</li>
        <li>For each citing case, view decision date, citation, court, jurisdiction, docket number, treatment, rationale, and URL.</li>
      </ul>
      You can also:
      <ul>
        <li>Use <strong>“Export to CSV”</strong> to download all citing cases.</li>
        <li>Use <strong>“Show more”</strong> to load additional citing cases when the list is long.</li>
      </ul>
    </li>
  </ol>

  <h3>Chatbot</h3>
  <p>Use the <strong>Chatbot</strong> when you want to:</p>
  <ul>
    <li>Understand ADA concepts and doctrine</li>
    <li>Get summaries of specific cases</li>
    <li>See how a case has been treated by later decisions</li>
    <li>Compare how different cases handle the same ADA issue</li>
    <li>Explore patterns (for example, remote work as an accommodation)</li>
    <li>Ask “what if” scenarios for fact patterns</li>
  </ul>
  <ol>
    <li>
      <strong>Select “Chatbot” mode</strong><br/>
      At the top of the app, select <strong>“Chatbot”</strong>.
    </li>
    <li>
      <strong>Type a question in natural language</strong><br/>
      Use the chat input at the bottom (<em>“Ask me anything”</em>). The chatbot will answer using the ADA case database and knowledge graph.<br/>
      Each answer includes a <strong>Copy</strong> button so you can copy the full response.
    </li>
  </ol>

  <h3>Configure Case Labels: how to adjust evaluation settings</h3>
  <p>
    The <strong>“Configure Case Labels”</strong> settings let you control how the tool assigns
    the case-level labels <em>Good</em>, <em>Bad</em>, <em>Moderate</em>, and <em>Unknown</em>.
  </p>
  <p>You can adjust:</p>
  <ul>
    <li><strong>Proportion thresholds</strong> – how dominant a treatment (Positive, Negative, Neutral, Unknown) must be at a court level to drive the label.</li>
    <li><strong>Time-based weighting</strong> – how much more recent citations count compared to older ones.</li>
    <li><strong>Time window</strong> – the date range of citations that should influence the label.</li>
    <li><strong>Jurisdiction weights</strong> – extra weight for citations from selected jurisdictions.</li>
    <li><strong>Court selection strategy</strong> – whether to use only the highest court or “walk down” to lower courts when the signal is mixed.</li>
    <li><strong>Label priority order</strong> – tie-breaking order when more than one label passes its threshold.</li>
  </ul>
  <p>
    If you are not sure what to choose, you can keep the defaults. The default configuration aims
    to balance stability (not changing labels too easily) with sensitivity (capturing clear trends in the case law).
  </p>
  <p>Examples of when to use configuration:</p>
  <ul>
    <li>
      <strong>You want more cautious labels</strong>: increase the proportion thresholds so that a label
      only applies when its treatment share is very strong.
    </li>
    <li>
      <strong>You care more about recent law</strong>: increase the maximum time-based weight or narrow the time window
      so that older citations matter less.
    </li>
    <li>
      <strong>You want to focus on certain jurisdictions</strong>: add jurisdiction weights so that citations from those
      courts influence the labels more.
    </li>
  </ul>

  <h3>Example Chatbot questions</h3>

  <h4>Orientation / Capability Questions</h4>
  <ol>
    <li>What kinds of cases are included in your database?</li>
    <li>What can you help me with regarding U.S. disability law and the ADA?</li>
  </ol>

  <h4>Case-Specific Questions (Single Case)</h4>
  <ol start="3">
    <li>Give me a short summary of "Access Now, Inc. v. Southwest Airlines Co."</li>
    <li>How does "Guy Amir v. St. Louis University" interpret "essential job functions" under the ADA?</li>
  </ol>

  <h4>Citation Treatment and Label Questions</h4>
  <ol start="5">
    <li>Show me citing cases that treat "Guy Amir v. St. Louis University" positively and explain why.</li>
    <li>Show me citing cases that criticize or limit "Access Now, Inc. v. Southwest Airlines Co." and explain how.</li>
  </ol>

  <h4>Comparing Cases</h4>
  <ol start="7">
    <li>Compare [Case A] and [Case B] on how they define "major life activity."</li>
    <li>How do [Case A] and [Case B] differ on what counts as a reasonable accommodation?</li>
  </ol>

  <h4>ADA Concept / Doctrine Questions</h4>
  <ol start="9">
    <li>What are "major life activities" under the ADA? Give examples and leading cases.</li>
    <li>What does "qualified individual with a disability" mean?</li>
  </ol>

  <h4>Hypothetical / Scenario Questions</h4>
  <ol start="11">
    <li>
      I represent an employee with Type 1 diabetes who was fired after asking for flexible break times
      to manage their insulin. The employer says regular breaks already exist and no extra accommodation
      is needed. Based on the ‘Good’ cases in this tool, which precedents are most supportive of the
      employee’s position and why?
    </li>
    <li>
      I represent a hospital that denied a nurse’s request to work fully remote due to patient-care duties.
      The nurse claims this is disability discrimination. Using the ‘Bad’ cases (cases that are unfavorable
      to plaintiffs with similar claims), which precedents should I review that could strengthen the hospital’s defense?
    </li>
    <li>
      A city government employee with chronic back pain requests a partial work-from-home schedule and
      specialized ergonomic equipment. The city approved the equipment but denied remote work. Please suggest
      ‘Moderate’ cases with mixed or nuanced outcomes on similar accommodation requests, and explain how courts
      balanced the employee’s needs and the employer’s operational limits.
    </li>
  </ol>

  <h4>Research / Pattern-Finding Questions</h4>
  <ol start="14">
    <li>Show me ADA cases that deal with remote work as a reasonable accommodation.</li>
    <li>List leading ADA cases on mental health conditions, learning disabilities, chronic pain, or obesity.</li>
    <li>Give examples of cases where plaintiffs lost because they were not considered “qualified individuals.”</li>
  </ol>
</div>
"""


# ---------------------- Cypher Queries ----------------------

CASE_SEARCH_QUERY = """
MATCH (c:Case)
WHERE toLower(c.name) CONTAINS toLower($search_term)
RETURN c.id AS case_id,
       coalesce(c.name, '') AS name,
       coalesce(c.citation_pipe, '') AS citation_pipe,
       c.decision_date AS decision_date,
       coalesce(c.court_name, '') AS court_name,
       coalesce(c.jurisdiction_name, '') AS jurisdiction_name,
       coalesce(c.docket_number, '') AS docket_number,
       coalesce(c.opinion_summary, '') AS opinion_summary,
       coalesce(c.court_listener_url, '') AS court_listener_url
ORDER BY c.decision_date DESC
LIMIT 10
"""

CITATIONS_FOR_CASE_QUERY = """
MATCH (citing_case:Case)-[r:CITES_TO]->(cited_case:Case {id: $case_id})
OPTIONAL MATCH (citing_case)-[:HEARD_IN]->(ct:Court)
RETURN coalesce(citing_case.name, 'Unknown') AS name,
       citing_case.decision_date AS decision_date,
       coalesce(citing_case.citation_pipe, '') AS citation_pipe,
       citing_case.court_level_case_label_decision AS decision_level,
       coalesce(ct.name, citing_case.court_name, 'Unknown') AS court_name,
       coalesce(citing_case.jurisdiction_name, 'Unknown') AS jurisdiction_name,
       coalesce(citing_case.docket_number, '') AS docket_number,
       coalesce(citing_case.court_listener_url, '') AS court_listener_url,
       coalesce(r.treatment_label, 'Unknown') AS treatment_label,
       coalesce(r.treatment_rationale, 'No rationale') AS treatment_rationale
ORDER BY citing_case.court_name, citing_case.decision_date DESC
"""

CASE_LABEL_EVALUATION_QUERY = """
MATCH (c:Case {id: $case_id})
RETURN coalesce(c.case_label, '') AS case_label,
       c.court_level_case_label_decision AS decision_level,
       coalesce(c.label_rationale, '') AS label_rationale
"""

TIME_WINDOW_QUERY = """
MATCH (src:Case)-[:CITES_TO]->(:Case)
WHERE src.decision_date IS NOT NULL
RETURN DISTINCT src.decision_date AS decision_date
"""

CASE_COUNT_QUERY = """
MATCH (c:Case)
RETURN count(c) AS case_count
"""


# ---------------------- Neo4j Client ----------------------

class Neo4jClient:
    def __init__(self, driver: Driver, database: str):
        self.driver = driver
        self.database = database

    def run_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.driver:
            return []
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params or {})
                return [record.data() for record in result]
        except Exception as e:
            st.error(f"Database query failed: {str(e)}")
            return []

    def search_cases(self, search_term: str) -> List[Dict[str, Any]]:
        search_term = search_term.strip()
        if not search_term:
            return []
        return self.run_query(CASE_SEARCH_QUERY, {"search_term": search_term})

    def get_citations_for_case(self, case_id: int) -> Dict[str, Any]:
        results = self.run_query(CITATIONS_FOR_CASE_QUERY, {"case_id": case_id})
        return {"count": len(results), "citations": results}

    def get_case_label_evaluation(self, case_id: int) -> Optional[Dict[str, Any]]:
        results = self.run_query(CASE_LABEL_EVALUATION_QUERY, {"case_id": case_id})
        if results:
            return results[0]
        return None

    def get_case_count(self) -> Optional[int]:
        rows = self.run_query(CASE_COUNT_QUERY, {})
        if rows:
            return rows[0].get("case_count")
        return None


# ---------------------- Initialization ----------------------

@st.cache_resource
def init_neo4j() -> Tuple[Optional[Neo4jClient], Optional[str]]:
    try:
        uri = st.secrets["NEO4J_URI"]
        user = st.secrets["NEO4J_USER"]
        password = st.secrets["NEO4J_PASSWORD"]
        database = st.secrets.get("NEO4J_DATABASE", "neo4j")

        driver = GraphDatabase.driver(uri, auth=(user, password))
        client = Neo4jClient(driver, database)
        return client, database
    except Exception as e:
        st.error(f"Failed to initialize Neo4j connection: {str(e)}")
        return None, None


# ---------------------- Helpers ----------------------

def format_date(date_value: Any) -> str:
    if not date_value:
        return "Unknown date"
    return str(date_value)[:10]


def _to_pydate(val: Any) -> Optional[date]:
    if val is None:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    if hasattr(val, "to_native"):
        try:
            native = val.to_native()
            if isinstance(native, date) and not isinstance(native, datetime):
                return native
            if isinstance(native, datetime):
                return native.date()
        except Exception:
            pass
    try:
        text = str(val)
        if "T" in text:
            return datetime.fromisoformat(text).date()
        return datetime.fromisoformat(text).date()
    except Exception:
        return None


def compute_time_window_defaults(client: Optional[Neo4jClient]) -> Tuple[date, date]:
    fallback_start = date(1990, 1, 1)
    fallback_end = date.today()
    if not client or not client.driver:
        return fallback_start, fallback_end

    try:
        with client.driver.session(database=client.database) as session:
            rows = session.run(TIME_WINDOW_QUERY, {}).data()
        ordinals: List[int] = []
        for row in rows:
            d = _to_pydate(row.get("decision_date"))
            if d is not None:
                ordinals.append(d.toordinal())
        if not ordinals:
            return fallback_start, fallback_end
        ordinals.sort()
        n = len(ordinals)
        if n == 1:
            q_idx = 0
        else:
            q_pos = 0.25 * (n - 1)
            q_idx = int(round(q_pos))
            q_idx = max(0, min(q_idx, n - 1))
        tmin = date.fromordinal(ordinals[q_idx])
        tmax = date.fromordinal(ordinals[-1])
        return tmin, tmax
    except Exception:
        return fallback_start, fallback_end


def get_treatment_color(treatment_label: str) -> str:
    if not treatment_label:
        return "#cccccc"
    treatment_lower = treatment_label.lower()

    negative_keywords = ["negative", "NEGATIVE"]
    if any(keyword.lower() in treatment_lower for keyword in negative_keywords):
        return "#e74c3c"

    positive_keywords = ["positive", "POSITIVE"]
    if any(keyword.lower() in treatment_lower for keyword in positive_keywords):
        return "#2ecc71"

    neutral_keywords = ["neutral", "NEUTRAL"]
    if any(keyword.lower() in treatment_lower for keyword in neutral_keywords):
        return "#fbc02d"

    ambiguous_keywords = ["ambiguous", "AMBIGUOUS"]
    if any(keyword.lower() in treatment_lower for keyword in ambiguous_keywords):
        return "#3498db"

    return "#cccccc"


def get_case_label_color(case_label: str) -> str:
    if not case_label:
        return "#cccccc"
    label_lower = case_label.lower()

    if label_lower == "good":
        return "#2ecc71"
    if label_lower == "bad":
        return "#e74c3c"
    if label_lower == "moderate":
        return "#fbc02d"
    if label_lower == "unknown":
        return "#3498db"
    return "#cccccc"


def render_citation_evaluation(client: Neo4jClient, case_id: int) -> None:
    evaluation = client.get_case_label_evaluation(case_id)

    if not evaluation:
        st.write("**Citation Evaluation:** *No evaluation available.*")
        return

    case_label = evaluation.get("case_label", "")
    decision_level = evaluation.get("decision_level")
    rationale = evaluation.get("label_rationale", "")

    if not case_label:
        st.write("**Citation Evaluation:** *No evaluation available.*")
        return

    label_color = get_case_label_color(case_label)
    label_display = (
        f'<span style="color: {label_color}; font-weight: 500;">{case_label}</span>'
    )

    st.markdown(
        f"**Citation Evaluation:** {label_display}", unsafe_allow_html=True
    )

    if decision_level is not None:
        st.write(f"**Decision Level:** {decision_level}")
    else:
        st.write("**Decision Level:** *Not specified*")

    if rationale:
        st.markdown(
            f"**Rationale:**<br>{rationale}", unsafe_allow_html=True
        )
    else:
        st.write("**Rationale:** *No rationale available.*")

    with st.expander("Methodology"):
        st.markdown(
            """
            The final label is determined by court level and time-weighted citation history.
            At each court level, the method computes the share of positive, negative, neutral,
            and unknown treatments with each citation weighted by its
            decision date and, optionally, its jurisdiction. A label can drive the result
            at a court only when its share passes a configured threshold. If no label
            passes at the highest court, the method can walk down to lower courts that
            provide a clearer signal (if the "Walk Down" Court Selection Strategy is enabled).
            """,
            unsafe_allow_html=True,
        )


def render_citations_dropdowns(citing_cases: List[Dict[str, Any]]) -> None:
    if not citing_cases:
        st.write("*No citing cases found.*")
        return

    _inject_citation_styles()

    for case in citing_cases:
        _render_single_citation(case)


def _inject_citation_styles() -> None:
    css = """
    <style>
        details.citation-card {
            margin-bottom: 12px !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            background: white !important;
            font-family: Arial, Helvetica, sans-serif !important;
        }

        details.citation-card summary {
            display: flex !important;
            align-items: center !important;
            padding: 12px 16px !important;
            background-color: #f8f9fa !important;
            cursor: pointer !important;
            user-select: none !important;
            transition: background-color 0.2s ease !important;
            list-style: none !important;
            font-family: Arial, Helvetica, sans-serif !important;
        }

        details.citation-card summary::-webkit-details-marker {
            display: none !important;
        }

        details.citation-card summary:hover {
            background-color: #f1f3f4 !important;
        }

        details.citation-card .citation-arrow {
            font-size: 14px !important;
            margin-left: auto !important;
            margin-right: 0 !important;
            transition: transform 0.2s ease !important;
            color: #666 !important;
            font-family: Arial, Helvetica, sans-serif !important;
        }

        details.citation-card .citation-indicator {
            width: 12px !important;
            height: 12px !important;
            border-radius: 50% !important;
            margin-right: 10px !important;
            flex-shrink: 0 !important;
            display: inline-block !important;
        }

        details.citation-card .citation-title {
            font-weight: 500 !important;
            color: #1f2937 !important;
            flex-grow: 1 !important;
            font-family: Arial, Helvetica, sans-serif !important;
        }

        details.citation-card .citation-content {
            padding: 16px !important;
            border-top: 1px solid #e0e0e0 !important;
            background: white !important;
            font-family: Arial, Helvetica, sans-serif !important;
        }

        details.citation-card .citation-field {
            margin-bottom: 8px !important;
            font-family: Arial, Helvetica, sans-serif !important;
        }

        details.citation-card .citation-label {
            font-weight: 600 !important;
            color: #374151 !important;
            margin-right: 8px !important;
            font-family: Arial, Helvetica, sans-serif !important;
        }

        details.citation-card[open] .citation-arrow {
            transform: rotate(-90deg) !important;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def _extract_case_data(case: Dict[str, Any]) -> Dict[str, str]:
    url = case.get("court_listener_url", "")

    return {
        "name": case.get("name", "Unknown"),
        "date": format_date(case.get("decision_date")),
        "citation": case.get("citation_pipe", "") or "No citation",
        "court": case.get("court_name", "Unknown"),
        "jurisdiction": case.get("jurisdiction_name", "Unknown"),
        "docket_number": case.get("docket_number", "") or "No docket number",
        "treatment": case.get("treatment_label", "Unknown"),
        "rationale": case.get("treatment_rationale") or "No rationale provided",
        "color": get_treatment_color(case.get("treatment_label", "Unknown")),
        "link": (
            f"<a href='{url}' target='_blank'>{url}</a>" if url else "No URL available"
        ),
        "url_display": url if url else "No URL available",
    }


def _render_single_citation(case: Dict[str, Any]) -> None:
    case_data = _extract_case_data(case)

    name = case_data["name"].replace("<", "&lt;").replace(">", "&gt;")
    date_txt = case_data["date"].replace("<", "&lt;").replace(">", "&gt;")
    citation_txt = case_data["citation"].replace("<", "&lt;").replace(">", "&gt;")
    court = case_data["court"].replace("<", "&lt;").replace(">", "&gt;")
    jurisdiction = case_data["jurisdiction"].replace("<", "&lt;").replace(">", "&gt;")
    docket = case_data["docket_number"].replace("<", "&lt;").replace(">", "&gt;")
    treatment = case_data["treatment"].replace("<", "&lt;").replace(">", "&gt;")
    rationale = (
        case_data["rationale"]
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )

    html = f"""
    <details class="citation-card">
        <summary>
            <span class="citation-indicator" style="background-color: {case_data['color']};"></span>
            <span class="citation-title">{name}</span>
            <span class="citation-arrow">◀</span>
        </summary>
        <div class="citation-content">
            <div class="citation-field">
                <span class="citation-label">Decision Date:</span>{date_txt}
            </div>
            <div class="citation-field">
                <span class="citation-label">Citation:</span>{citation_txt}
            </div>
            <div class="citation-field">
                <span class="citation-label">Court:</span>{court}
            </div>
            <div class="citation-field">
                <span class="citation-label">Jurisdiction:</span>{jurisdiction}
            </div>
            <div class="citation-field">
                <span class="citation-label">Docket Number:</span>{docket}
            </div>
            <div class="citation-field">
                <span class="citation-label">Treatment:</span>{treatment}
            </div>
            <div class="citation-field">
                <span class="citation-label">Rationale:</span><br>{rationale}
            </div>
            <div class="citation-field">
                <span class="citation-label">URL:</span>{case_data['link']}
            </div>
        </div>
    </details>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_case(client: Neo4jClient, case: Dict[str, Any]) -> None:
    case_id = case.get("case_id")
    name = case.get("name", "Unknown")
    citation = case.get("citation_pipe", "No citation")
    decision = format_date(case.get("decision_date"))
    court = case.get("court_name", "Unknown court")
    jurisdiction = case.get("jurisdiction_name", "Unknown jurisdiction")
    docket_number = case.get("docket_number", "") or "No docket number"
    summary = case.get("opinion_summary", "No summary available")
    url = case.get("court_listener_url", "")

    st.write(f"**Name:** {name}")
    st.write(f"**Citation:** {citation}")
    st.write(f"**Decision Date:** {decision}")
    st.write(f"**Court:** {court}")
    st.write(f"**Jurisdiction:** {jurisdiction}")
    st.write(f"**Docket Number:** {docket_number}")
    st.write("**Summary:**")
    st.write(summary)
    st.write(f"**URL:** {url or 'No URL available'}")

    st.write("---")
    render_citation_evaluation(client, case_id)
    st.write("---")

    citations_data = client.get_citations_for_case(case_id)
    citations_count = citations_data["count"]
    citing_cases = citations_data["citations"]

    csv_bytes: Optional[bytes] = None
    if citations_count > 0:
        rows_for_csv = []
        for c in citing_cases:
            rows_for_csv.append(
                {
                    "Citation Name": c.get("name", ""),
                    "Decision Date": format_date(c.get("decision_date")),
                    "Citation": c.get("citation_pipe", ""),
                    "Court": c.get("court_name", ""),
                    "Jurisdiction": c.get("jurisdiction_name", ""),
                    "Docket Number": c.get("docket_number", ""),
                    "Treatment": c.get("treatment_label", ""),
                    "Rationale": c.get("treatment_rationale", ""),
                    "URL": c.get("court_listener_url", ""),
                }
            )
        df = pd.DataFrame(rows_for_csv)
        csv_bytes = df.to_csv(index=False).encode("utf-8")

    st.write(f"**Citing Cases ({citations_count}):**")

    citations_container = st.container()
    footer_container = st.container()

    page_key = f"citations_shown_{case_id}"
    if st.session_state.get("current_case_id") != case_id:
        st.session_state["current_case_id"] = case_id
        st.session_state[page_key] = CITATION_PAGE_SIZE

    if page_key not in st.session_state:
        st.session_state[page_key] = CITATION_PAGE_SIZE

    shown = st.session_state[page_key]
    if shown > citations_count:
        shown = citations_count
        st.session_state[page_key] = shown

    with footer_container:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            if csv_bytes is not None:
                st.download_button(
                    "Export to CSV",
                    data=csv_bytes,
                    file_name=f"citing_cases_{case_id}.csv",
                    mime="text/csv",
                    key=f"download_citations_{case_id}",
                )
        with col_right:
            st.write(f"{shown} out of {citations_count} citations")
            if shown < citations_count:
                show_more = st.button(
                    "Show more",
                    key=f"show_more_{case_id}",
                )
                if show_more:
                    shown = min(shown + CITATION_PAGE_SIZE, citations_count)
                    st.session_state[page_key] = shown

    visible_cases = citing_cases[:shown]
    with citations_container:
        render_citations_dropdowns(visible_cases)

    st.write("---")


def render_connection_help() -> None:
    st.error("❌ Database connection failed. Please check your Neo4j credentials.")
    st.info("Make sure your Streamlit secrets are configured with:")
    st.code(
        """
NEO4J_URI = "your_neo4j_uri"
NEO4J_USER = "your_username"
NEO4J_PASSWORD = "your_password"
NEO4J_DATABASE = "neo4j"
        """
    )


def render_copy_button(text: str) -> None:
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")

    html = f"""
    <html>
      <body style="margin:0; padding:0;">
        <button
          type="button"
          data-copy="{b64}"
          onclick="navigator.clipboard.writeText(atob(this.dataset.copy))"
          title="Copy to clipboard"
          style="
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.25rem 0.75rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(49, 51, 63, 0.2);
            background-color: rgb(240, 242, 246);
            color: rgb(49, 51, 63);
            font-size: 0.875rem;
            font-weight: 400;
            cursor: pointer;
          "
        >
          Copy
        </button>
      </body>
    </html>
    """

    components.html(html, height=40, width=90)


# ---------------------- Chatbot retry helper ----------------------

def call_chatbot_with_retries(
    prompt: str,
    max_attempts: int = 3,
    delay_seconds: float = 5.0,
) -> str:
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return invoke_agent_text(multi_tool_agent, prompt)
        except Exception as e:  # noqa: BLE001
            last_error = e
            message = str(e).lower()
            is_throttle = any(
                key in message
                for key in [
                    "throttle",
                    "rate limit",
                    "rate-limit",
                    "too many requests",
                    "429",
                ]
            )

            if is_throttle and attempt < max_attempts:
                try:
                    st.toast("Still thinking...")
                except Exception:
                    st.info("Still thinking...")
                time.sleep(delay_seconds)
                continue
            break

    if last_error is not None:
        print(f"Chatbot call failed after retries: {last_error}")

    return "Sorry, something went wrong on our side while generating this answer. Please try again."


# ---------------------- Case label configuration UI ----------------------

def render_case_label_configuration(client: Optional[Neo4jClient]) -> None:
    st.markdown(
        """
This section lets you configure how case-level labels (**Good**, **Bad**, **Moderate**, and
**Unknown**) are computed from citation history. You can adjust thresholds, the strength
of time-based weighting, jurisdiction preferences, and how courts are used in the
decision rule.  
For more information on these options, please see the detailed guide
[here](https://drive.google.com/file/d/1Q-AASvyen7ElBbY9m2mvy3qFXjStdfR8/view?usp=sharing).
""",
        unsafe_allow_html=True,
    )

    default_tmin, default_tmax = compute_time_window_defaults(client)

    if "cfg_tmin_date" not in st.session_state:
        st.session_state["cfg_tmin_date"] = default_tmin
    if "cfg_tmax_date" not in st.session_state:
        st.session_state["cfg_tmax_date"] = default_tmax

    # ---------- 1. Proportion Thresholds ----------
    st.markdown("### 1. Proportion Thresholds")

    col_input, col_purpose, col_effect = st.columns([3, 3, 4])

    with col_input:
        prev_default_thresholds = st.session_state.get(
            "cfg_default_thresholds", True
        )
        use_default_thresholds = st.checkbox(
            "Default (0.55 for all labels)",
            value=prev_default_thresholds,
            key="cfg_default_thresholds",
        )
        if use_default_thresholds and not prev_default_thresholds:
            st.session_state["cfg_thr_pos"] = 0.55
            st.session_state["cfg_thr_neg"] = 0.55
            st.session_state["cfg_thr_neu"] = 0.55
            st.session_state["cfg_thr_unk"] = 0.55

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            thr_pos = st.number_input(
                "Positive",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
                key="cfg_thr_pos",
                disabled=use_default_thresholds,
            )
        with c2:
            thr_neg = st.number_input(
                "Negative",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
                key="cfg_thr_neg",
                disabled=use_default_thresholds,
            )
        with c3:
            thr_neu = st.number_input(
                "Neutral",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
                key="cfg_thr_neu",
                disabled=use_default_thresholds,
            )
        with c4:
            thr_unk = st.number_input(
                "Unknown",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
                key="cfg_thr_unk",
                disabled=use_default_thresholds,
            )

    with col_purpose:
        st.markdown(
            """
**Purpose**  
How large a share a label must have at a court level to count as dominant.
""",
            unsafe_allow_html=True,
        )

    with col_effect:
        st.markdown(
            """
**Effect**  
- Higher thresholds (for example, 0.70) make the method more cautious; more cases become *Moderate* or *Unknown*.  
- Lower thresholds (for example, 0.50) make it easier to call a case *Good* or *Bad*.
""",
            unsafe_allow_html=True,
        )

    # ---------- 2. Maximum Time-Based Weight ----------
    st.markdown("---")
    st.markdown("### 2. Maximum Time-Based Weight")

    col_input, col_purpose, col_effect = st.columns([3, 3, 4])
    with col_input:
        prev_default_max_weight = st.session_state.get(
            "cfg_default_max_weight", True
        )
        use_default_max_weight = st.checkbox(
            "Default (MAX_WEIGHT = 2.5)",
            value=prev_default_max_weight,
            key="cfg_default_max_weight",
        )
        if use_default_max_weight and not prev_default_max_weight:
            st.session_state["cfg_max_weight"] = 2.5

        max_weight_value = st.number_input(
            "MAX_WEIGHT",
            min_value=1.0,
            value=2.5,
            step=0.1,
            key="cfg_max_weight",
            disabled=use_default_max_weight,
        )

    with col_purpose:
        st.markdown(
            """
**Purpose**  
How much more recent citations should count than older ones.
""",
            unsafe_allow_html=True,
        )

    with col_effect:
        st.markdown(
            """
**Effect**  
- Larger values (for example, 4.0) give strong preference to recent citations.  
- Smaller values (for example, 1.5) reduce the gap between old and recent citations.
""",
            unsafe_allow_html=True,
        )

    # ---------- 3. Time Window ----------
    st.markdown("---")
    st.markdown("### 3. Time Window")

    col_input, col_purpose, col_effect = st.columns([3, 3, 4])

    with col_input:
        prev_default_time_window = st.session_state.get(
            "cfg_default_time_window", True
        )
        use_default_time_window = st.checkbox(
            "Default (tmin = First Quartile (Q1), tmax = Latest Citation Decision of the current database)",
            value=prev_default_time_window,
            key="cfg_default_time_window",
        )

        if use_default_time_window and not prev_default_time_window:
            st.session_state["cfg_tmin_date"] = default_tmin
            st.session_state["cfg_tmax_date"] = default_tmax

        c1, c2 = st.columns(2)
        with c1:
            tmin_date = st.date_input(
                "Start date (tmin)",
                value=default_tmin,
                key="cfg_tmin_date",
                disabled=use_default_time_window,
            )
        with c2:
            tmax_date = st.date_input(
                "End date (tmax)",
                value=default_tmax,
                key="cfg_tmax_date",
                disabled=use_default_time_window,
            )

    with col_purpose:
        st.markdown(
            """
**Purpose**  
Define the period over which recency is measured.
""",
            unsafe_allow_html=True,
        )

    with col_effect:
        st.markdown(
            """
**Effect**  
- Narrow windows focus strongly on very recent law.  
- Wider windows allow older citations to retain more weight.
""",
            unsafe_allow_html=True,
        )

    # ---------- 4. Jurisdiction Weights ----------
    st.markdown("---")
    st.markdown("### 4. Jurisdiction Weights")

    col_input, col_purpose, col_effect = st.columns([3, 3, 4])

    with col_input:
        prev_default_juris = st.session_state.get("cfg_default_juris", True)
        use_default_juris = st.checkbox(
            "Default (no extra jurisdiction weights)",
            value=prev_default_juris,
            key="cfg_default_juris",
        )

        if "cfg_juris_count" not in st.session_state:
            st.session_state["cfg_juris_count"] = 0

        if use_default_juris and not prev_default_juris:
            st.session_state["cfg_juris_count"] = 0

        if not use_default_juris:
            add_col, _ = st.columns([1, 5])
            with add_col:
                can_add_more = (
                    st.session_state["cfg_juris_count"] < len(VALID_JURIS_LIST)
                )
                if st.button(
                    "Add jurisdiction",
                    key="cfg_add_jurisdiction",
                    disabled=not can_add_more,
                ):
                    if can_add_more:
                        st.session_state["cfg_juris_count"] += 1

            for i in range(st.session_state["cfg_juris_count"]):
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.selectbox(
                        "Jurisdiction",
                        VALID_JURIS_LIST,
                        key=f"cfg_juris_name_{i}",
                    )
                with c2:
                    st.number_input(
                        "Weight (Jᵢ)",
                        min_value=0.0,
                        value=1.0,
                        step=0.5,
                        key=f"cfg_juris_weight_{i}",
                    )
        else:
            st.caption("No jurisdictions have extra weight under the default setting.")

    with col_purpose:
        st.markdown(
            """
**Purpose**  
Give extra influence to citations from chosen jurisdictions.
""",
            unsafe_allow_html=True,
        )

    with col_effect:
        st.markdown(
            """
**Effect**  
Jurisdictions with higher Ji make their citations count more toward the final label.
""",
            unsafe_allow_html=True,
        )

    # ---------- 5. Court Selection Strategy ----------
    st.markdown("---")
    st.markdown("### 5. Court Selection Strategy")

    col_input, col_purpose, col_effect = st.columns([3, 3, 4])

    with col_input:
        court_strategy = st.radio(
            "Strategy",
            options=["Walk Down (default)", "Highest Court Only"],
            index=0,
            horizontal=True,
            key="cfg_court_strategy",
        )

    with col_purpose:
        st.markdown(
            """
**Purpose**  
Decide whether only the highest court matters, or whether lower courts can drive
the label when the highest court is mixed.
""",
            unsafe_allow_html=True,
        )

    with col_effect:
        st.markdown(
            """
**Effect**  
- **Highest Court Only**: strict focus on the top court that cites the case.  
- **Walk Down**: if the highest court is too mixed, lower courts with clearer signals
  may decide the label.
""",
            unsafe_allow_html=True,
        )

    # ---------- 6. Label Priority Order ----------
    st.markdown("---")
    st.markdown("### 6. Label Priority Order")

    col_input, col_purpose, col_effect = st.columns([3, 3, 4])

    label_options = ["Unknown", "Negative", "Neutral", "Positive"]
    label_to_token = {
        "Positive": "pos",
        "Negative": "neg",
        "Neutral": "neu",
        "Unknown": "unk",
    }

    with col_input:
        prev_default_priority = st.session_state.get(
            "cfg_default_priority", True
        )
        use_default_priority = st.checkbox(
            "Default (1. Unknown, 2. Negative, 3. Neutral, 4. Positive)",
            value=prev_default_priority,
            key="cfg_default_priority",
        )

        if use_default_priority and not prev_default_priority:
            st.session_state["cfg_prio_1"] = "Unknown"
            st.session_state["cfg_prio_2"] = "Negative"
            st.session_state["cfg_prio_3"] = "Neutral"
            st.session_state["cfg_prio_4"] = "Positive"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            prio_1 = st.selectbox(
                "1.",
                label_options,
                index=0,
                key="cfg_prio_1",
                disabled=use_default_priority,
            )
        with c2:
            prio_2 = st.selectbox(
                "2.",
                label_options,
                index=1,
                key="cfg_prio_2",
                disabled=use_default_priority,
            )
        with c3:
            prio_3 = st.selectbox(
                "3.",
                label_options,
                index=2,
                key="cfg_prio_3",
                disabled=use_default_priority,
            )
        with c4:
            prio_4 = st.selectbox(
                "4.",
                label_options,
                index=3,
                key="cfg_prio_4",
                disabled=use_default_priority,
            )

    with col_purpose:
        st.markdown(
            """
**Purpose**  
Break ties when more than one label passes its threshold at the same court level.
""",
            unsafe_allow_html=True,
        )

    with col_effect:
        st.markdown(
            """
**Effect**  
- Placing **Negative** earlier makes the method more pessimistic.  
- Placing **Positive** earlier makes it more optimistic.
""",
            unsafe_allow_html=True,
        )

    # ---------- Update Case Labels Button ----------

    st.markdown("---")
    update_clicked = st.button("Update Case Labels", key="cfg_update_case_labels")

    if not update_clicked:
        return

    can_run = True

    label_thresholds: Optional[Dict[str, float]] = None
    if not use_default_thresholds:
        label_thresholds = {
            "Pos_p": float(thr_pos),
            "Neg_p": float(thr_neg),
            "Neu_p": float(thr_neu),
            "Unk_p": float(thr_unk),
        }

    default_time_weight = use_default_max_weight
    time_weight = None
    if not use_default_max_weight:
        if max_weight_value < 1.0:
            st.error("MAX_WEIGHT must be at least 1.0.")
            can_run = False
        else:
            time_weight = [1.0, float(max_weight_value)]

    default_tmin_tmax = use_default_time_window
    tmin_tmax = None
    if not use_default_time_window:
        if tmax_date <= tmin_date:
            st.error("End date (tmax) must be after start date (tmin).")
            can_run = False
        else:
            tmin_tmax = [tmin_date, tmax_date]

    jurisdictions_param: Optional[Dict[str, float]] = None
    if not use_default_juris:
        jurisdictions_param = {}
        count = st.session_state.get("cfg_juris_count", 0)
        for i in range(count):
            name = st.session_state.get(f"cfg_juris_name_{i}")
            weight = st.session_state.get(f"cfg_juris_weight_{i}", 0.0)
            if name and name in VALID_JURIS_LIST:
                jurisdictions_param[name] = float(weight)

    lower_level_court = court_strategy.startswith("Walk Down")

    default_label_priority = use_default_priority
    label_priority_param: Optional[List[str]] = None
    if not use_default_priority:
        chosen_labels = [prio_1, prio_2, prio_3, prio_4]
        if len(set(chosen_labels)) != 4:
            st.error(
                "Each label must appear exactly once in the priority order."
            )
            can_run = False
        else:
            label_priority_param = [label_to_token[l] for l in chosen_labels]

    if not can_run:
        return

    try:
        with st.spinner(
            "Updating case labels with the current configuration."
        ):
            label_all_cases(
                force=True,
                echo=True,
                lower_level_court=lower_level_court,
                include_unknown=True,
                label_thresholds=label_thresholds,
                default_label_priority=default_label_priority,
                label_priority=label_priority_param,
                default_tmin_tmax=default_tmin_tmax,
                tmin_tmax=tmin_tmax,
                default_time_weight=default_time_weight,
                time_weight=time_weight,
                non_linear_recency_effect=False,
                jurisdictions=jurisdictions_param,
                results_csv=False,
            )
        st.success(
            "Case labels have been updated using the current configuration."
        )
    except Exception as e:
        st.error(f"Case labeling failed: {e}")


# ---------------------- Streamlit UI ----------------------

st.set_page_config(
    page_title="Legal Citation Machine",
    layout="wide",
)

st.markdown(
    """
<style>
    html, body, [class*="css"] {
        font-family: Arial, Helvetica, sans-serif !important;
    }
    .stApp {
        font-family: Arial, Helvetica, sans-serif !important;
    }
    .stApp p, .stApp div:not([class*="icon"]):not([data-testid*="Icon"]]),
    .stApp span:not([class*="icon"]):not([data-testid*="Icon"]):not(.material-icons),
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp li, .stApp td, .stApp th, .stApp label, .stApp input,
    .stApp textarea, .stApp select, .stApp button {
        font-family: Arial, Helvetica, sans-serif !important;
    }

    .main .block-container {
        padding-top: 3.5rem;
        max-width: 1100px;
    }
    .main-title {
        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 2.5rem;
        font-weight: 300;
        color: #333;
        text-align: center;
        margin-bottom: 1rem;
    }

    .stTextInput > div > div {
        background: transparent !important;
        border: none !important;
    }
    .stTextInput > div > div > input {
        font-family: Arial, Helvetica, sans-serif !important;
        border: 1px solid #ddd !important;
        padding: 12px !important;
        font-size: 16px !important;
        background-color: white !important;
        width: 100% !important;
    }
    .stTextInput > label {
        display: none !important;
    }

    .streamlit-expanderHeader [data-testid="stExpanderToggleIcon"],
    .streamlit-expanderHeader [data-testid="stExpanderToggleIcon"] *,
    .streamlit-expanderHeader .material-icons,
    .streamlit-expanderHeader svg,
    .streamlit-expanderHeader svg * {
        font-family: 'Material Icons', 'MaterialIcons', 'Material Icons Outlined', 'Material Symbols Rounded' !important;
    }
    .streamlit-expanderHeader svg {
        width: 1.25rem !important;
        height: 1.25rem !important;
        margin-right: 6px !important;
    }

    /* Make all st.button widgets look like the Copy button */
    div.stButton > button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
        background-color: rgb(240, 242, 246);
        color: rgb(49, 51, 63);
        font-size: 0.875rem;
        font-weight: 400;
        cursor: pointer;
    }

    div[data-testid="column"] {
        padding-left: 0.25rem !important;
        padding-right: 0.25rem !important;
    }

    /* Help panel styling inside expander */
    .help-panel {
        border-radius: 12px;
        background-color: #ffffff;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
        padding: 1.5rem 1.75rem;
        margin-top: 0.25rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .help-panel h2 {
        margin-top: 0;
        margin-bottom: 0.75rem;
    }
    .help-panel h3 {
        margin-top: 1.25rem;
    }
    .help-panel h4 {
        margin-top: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="main-title">Legal Citation Machine</h1>',
    unsafe_allow_html=True,
)

# ---------------------- Session state for mode ----------------------

if "mode" not in st.session_state:
    st.session_state["mode"] = "case_lookup"

prev_mode = st.session_state["mode"]

# ---------------------- DB init ----------------------

client, _ = init_neo4j()

# ---------------------- Mode toggle (centered under title) ----------------------

# Use a narrow center column so the radio itself appears centered
left_col, center_col, right_col = st.columns([3, 1, 3])

with center_col:
    st.markdown(
        '<div style="text-align:center; font-weight:500; margin-bottom:0.25rem;">Mode</div>',
        unsafe_allow_html=True,
    )
    selected_option = st.radio(
        "",
        options=["Case Lookup", "Chatbot"],
        index=0 if prev_mode == "case_lookup" else 1,
        horizontal=True,
        key="mode_select",
        label_visibility="collapsed",
    )

mode = "case_lookup" if selected_option == "Case Lookup" else "chatbot"

if mode != prev_mode and mode == "case_lookup":
    st.session_state["chat_history"] = []
    st.session_state["last_chat_prompt"] = ""

st.session_state["mode"] = mode

# ---------------------- Intro text in original caption style ----------------------

st.markdown("---")

if client:
    case_count = client.get_case_count()
else:
    case_count = None

if case_count is not None:
    st.caption(
        f"""
*Legal Citation Machine is a UC Berkeley student project that uses a Neo4j knowledge graph of {case_count:,} ADA-related cases. It can (1) evaluate how a case is treated in later decisions and (2) answer natural-language questions about these cases, their citation history, and other information stored in the Neo4j database.*
"""
    )
else:
    st.caption(
        """
*This tool was built by a group of students at UC Berkeley to evaluate legal case
citations in the Americans with Disabilities Act (ADA) domain. ADA-relevant
cases are covered in the built-in knowledge graph. The evaluation result is based on
the historical treatment of the case within that graph.*
"""
    )

st.markdown("---")

# ---------------------- Expanders (now below the horizontal line) ----------------------

with st.expander("What this app does"):
    st.markdown(HELP_PANEL_HTML, unsafe_allow_html=True)

with st.expander("Configure Case Labels"):
    if client is None:
        st.error("Case label configuration requires a working Neo4j connection.")
    else:
        render_case_label_configuration(client)

# ---------------------- Case Lookup helper ----------------------

def case_search_function(searchterm: str) -> List[str]:
    if not client:
        st.session_state["case_search_options"] = {}
        return []

    if not searchterm or not searchterm.strip():
        st.session_state["case_search_options"] = {}
        return []

    matches = client.search_cases(searchterm)
    options_: Dict[str, Dict[str, Any]] = {}
    labels_: List[str] = []

    for m in matches:
        name_ = m.get("name", "Unknown")
        citation_ = m.get("citation_pipe", "")
        label_ = f"{name_} | {citation_}" if citation_ else name_
        if label_ in options_:
            label_ = f"{label_} [{m.get('case_id')}]"
        options_[label_] = m
        labels_.append(label_)

    st.session_state["case_search_options"] = options_
    return labels_


# ---------------------- Main mode switch (search / chat area) ----------------------

if not client:
    render_connection_help()
else:
    if st.session_state["mode"] == "case_lookup":
        # -------- Case Lookup UI --------
        selected_label = st_searchbox(
            case_search_function,
            placeholder="Search for a Case",
            key="case_search_box",
        )

        selected_case = None
        if selected_label:
            options = st.session_state.get("case_search_options", {})
            selected_case = options.get(selected_label)

        if selected_case:
            st.write("---")
            render_case(client, selected_case)

    else:
        # -------- Chatbot UI --------

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if "last_chat_prompt" not in st.session_state:
            st.session_state["last_chat_prompt"] = ""

        for idx, turn in enumerate(st.session_state["chat_history"]):
            with st.chat_message("user"):
                st.markdown(turn.get("user", ""))

            with st.chat_message("assistant"):
                st.markdown(turn.get("assistant", ""))

                col_copy, _ = st.columns([0.12, 0.88])
                with col_copy:
                    render_copy_button(turn.get("assistant", ""))

        user_prompt = st.chat_input("Ask me anything")

        if user_prompt:
            with st.spinner("Thinking..."):
                answer = call_chatbot_with_retries(user_prompt)

            st.session_state["chat_history"].append(
                {"user": user_prompt, "assistant": answer}
            )
            st.session_state["last_chat_prompt"] = user_prompt

            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
