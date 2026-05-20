import os
import json
import ast
import re
import asyncio
import logging
import warnings
import contextlib
import hashlib
from collections import defaultdict
from cachetools import TTLCache
from typing import TypedDict

# FORCE SUPPRESS ALL NOISY LOGS
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from scrapegraph_py import ScrapeGraphAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from tree_sitter_language_pack import get_parser
except ImportError:
    get_parser = None

from analysis_policy import (
    PRIORITY_FILES,
    PRIORITY_DIRS,
    EXCLUDE_DIRS,
    MAX_RECURSIVE_DEPTH,
    MAX_FILES_ANALYZED,
    MAX_FILE_LINES,
    TOP_K_PER_CATEGORY,
    MAX_RAG_K,
    MAX_CHUNKS_PER_SOURCE,
    BOILERPLATE_PHRASES,
    is_priority_path,
    should_exclude_path,
)

load_dotenv(".env")

# Groq / context limits — keep inputs and outputs small to avoid API errors
MAX_LLM_TOKENS = int(os.getenv("MAX_LLM_TOKENS", "1024"))
MAX_STATE_FIELD_CHARS = int(os.getenv("MAX_STATE_FIELD_CHARS", "1200"))
MAX_GRAPH_RELATIONS = int(os.getenv("MAX_GRAPH_RELATIONS", "25"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "350"))
MAX_RAG_K = int(os.getenv("MAX_RAG_K", "2"))


def normalize_repo(repo: str) -> str:
    """
    Normalize repository names consistently.
    Example:
    ScrapeGraphAI/Scrapegraph-ai
    ->
    scrapegraphai/scrapegraph-ai
    """

    if not repo:
        return ""

    repo = str(repo).strip()

    if "/" not in repo:
        return repo.lower()

    owner, name = repo.split("/", 1)

    return f"{owner.lower()}/{name.lower()}"



def truncate_for_api(text: str, max_chars: int = MAX_STATE_FIELD_CHARS) -> str:
    """Trim text before passing to the next LLM call or storing in state."""
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


CONCISE_OUTPUT_RULE = """
OUTPUT LIMITS (required):
- Be brief: max 3 short bullet points per section.
- Total response under 400 words.
- No repetition; skip empty sections.
"""

# =========================================================
# CACHE WITH TTL (Using cachetools for better memory safety)
# =========================================================

# TTL: 1 Hour, Max Size: 500 URLs
_SCRAPE_CACHE = TTLCache(maxsize=500, ttl=3600)
_INDEXED_URLS: set = set()  # urls already embedded in ChromaDB
_URL_LOCKS = defaultdict(asyncio.Lock)  # Per-URL concurrency control

# =========================================================
# TELEMETRY & TRACING
# =========================================================

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "graphora")
    print(f"🔍 LangSmith tracing ON → {os.environ['LANGCHAIN_PROJECT']}")

def get_cached_content(url: str):
    """Retrieves cached content if available in TTLCache."""
    return _SCRAPE_CACHE.get(url)

def set_cached_content(url: str, content: str):
    """Sets content in TTLCache."""
    _SCRAPE_CACHE[url] = content

# =========================================================
# CLEANING ENGINE
# =========================================================

def clean_text(text: str) -> str:
    """
    Production-grade cleaning for technical documentation.
    Removes GitHub UI elements and strictly extracts technical content.
    """
    # 1. REMOVE GITHUB UI ELEMENTS (Navigation, sidebars, buttons)
    # This prevents the "Skip to main content", "Settings", etc. from being indexed
    UI_PATTERNS = [
        r"Skip to content",
        r"Search or jump to...",
        r"Pull requests",
        r"Issues",
        r"Marketplace",
        r"Explore",
        r"Sign (in|up)",
        r"You are (signed|logged) in",
        r"Appearance settings",
        r"Keyboard shortcuts",
        r"Copyright .* GitHub, Inc\.",
        r"Help",
        r"Support",
        r"Community",
        r"Pricing",
        r"Notifications",
        r"Dashboard",
        r"Settings",
        r"Code",
        r"Actions",
        r"Projects",
        r"Security",
        r"Insights",
    ]
    for pattern in UI_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 2. Extract content between likely main body markers if present (heuristic)

    # Remove excessive repeated links/buttons in lists (common in GH sidebars)
    text = re.sub(r"\n\* \[ \] .*", "", text)
    text = re.sub(r"\n\* \[x\] .*", "", text)

    # Remove Javascript snippets and CSS
    text = re.sub(r"<script.*?</script>", "", text, flags=re.DOTALL)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL)

    # Strip remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove JSON wrapper artifacts (common with ScrapeGraphAI SDK)
    text = re.sub(r'"markdown"\s*:\s*"', "", text)
    text = re.sub(r"\\n", "\n", text)
    text = re.sub(r"\\t", " ", text)

    # Collapse excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove common boilerplate and navigation noise
    FILLER_PHRASES = [
        "cookie", "privacy policy", "terms of service", "all rights reserved",
        "subscribe", "click here", "sign up", "login", "navigation", "footer",
        "back to top", "skip to main content", "search `ctrl`+`k`", "choose version",
        "unfortunately", "i can't provide", "i cannot provide", "i'm unable to"
    ]

    lines = []
    for line in text.splitlines():
        line = line.strip()

        # Skip empty or very short lines
        if not line or len(line) < 10: # Increased threshold to 10 to skip buttons
            continue

        # Skip navigation links (md format) - GH sidebars are full of these
        if line.startswith("* [") and "](http" in line:
            continue

        # Skip common fillers
        if any(p in line.lower() for p in FILLER_PHRASES):
            continue

        lines.append(line)

    cleaned = "\n".join(lines).strip()
    return cleaned




llm = ChatGroq(
    model_name="llama-3.1-8b-instant",  # Downgraded to 8b to strictly stay within Groq TPD limits
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_retries=5,
    max_tokens=MAX_LLM_TOKENS,
)

# =========================================================
# SCRAPEGRAPH AI
# =========================================================

# reads SGAI_API_KEY from env automatically
sgai = ScrapeGraphAI()

# =========================================================
# NEO4J
# =========================================================

ALLOWED_RELATIONS = {
    "USES",
    "CALLS",
    "DEPENDS_ON",
    "IMPLEMENTS",
    "PROVIDES",
    "EXTENDS",
    "IMPORTS",
    "COUPLED_WITH",
}

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)

def init_neo4j():
    """Initializes Neo4j with unique constraints and relationship indexing for performance."""
    try:
        with driver.session() as session:
            # unique constraint on Technology nodes
            session.run("CREATE CONSTRAINT technology_name IF NOT EXISTS FOR (t:Technology) REQUIRE t.name IS UNIQUE")

            # Relationship indexes require an explicit type (no wildcard pattern)
            for rel_type in ALLOWED_RELATIONS:
                session.run(
                    f"""
                    CREATE INDEX repo_rel_idx_{rel_type.lower()} IF NOT EXISTS
                    FOR ()-[r:{rel_type}]-()
                    ON (r.repo)
                    """
                )

            print("⛓️ Neo4j Constraints & Performance Indices Initialized")
    except Exception as e:
        print(f"⚠️ Neo4j Init Warning: {e}")

init_neo4j()

# =========================================================
# KNOWLEDGE GRAPH FUNCTIONS (GraphRAG Extensions)
# =========================================================

def fetch_neighbors(repo: str, nodes: list[str], depth: int = 1) -> list[dict]:
    """
    GraphRAG Core: Fetches neighboring nodes and relationships to expand local context.
    Traverses the graph to find what the specified nodes depend on or what depends on them.
    """
    try:
        if not nodes: return []

        with driver.session() as session:
            query = """
            MATCH (n:Technology)-[r]-(neighbor:Technology)
            WHERE n.name IN $nodes AND r.repo = $repo
            RETURN n.name AS origin, type(r) AS relation, neighbor.name AS target
            LIMIT 50
            """
            result = session.run(query, nodes=nodes, repo=normalize_repo(repo))

            expansion = []
            for row in result:
                expansion.append({
                    "origin": row["origin"],
                    "relation": row["relation"],
                    "target": row["target"]
                })

            return expansion
    except Exception as e:
        print(f"❌ GraphRAG Expansion Error: {e}")
        return []

# =========================================================
# CHROMADB (VECTOR DB)
# =========================================================

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="repos",  # Unified collection; repo filtering is done via metadata
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# =========================================================
# MCP SERVER
# =========================================================

github_server = StdioServerParameters(
    command="cmd",
    args=["/c", "npx", "-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")},
)

# =========================================================
# GRAPH STATE
# =========================================================


class RepoState(TypedDict):
    query: str
    repo: str
    docs_url: str
    repo_data: str
    architecture_data: str
    activity_data: str
    docs_data: str
    intelligence_data: str
    graph_data: str
    security_data: str
    roadmap_data: str
    final_report: str
    retrieved_chunks: list[dict]  # To store RAG chunks for UI
    errors: list[str]  # tracking to prevent cascading hallucinations


# =========================================================
# KNOWLEDGE GRAPH FUNCTIONS
# =========================================================

def clean_relation(relation: str) -> str:
    """Standardizes relationship names for Neo4j."""
    rel = relation.upper().replace(" ", "_").replace("-", "_").replace("/", "_")
    if rel in ALLOWED_RELATIONS:
        return rel
    raise ValueError(f"Invalid relationship type: {relation}. Allowed types: {ALLOWED_RELATIONS}")


def store_relationship(
    source: str,
    relation: str,
    target: str,
    repo: str,
    confidence: float = 1.0,
) -> None:
    """
    Stores a grounded relationship in Neo4j Knowledge Graph.
    Includes deduplication + validation + self-loop protection.
    """

    try:

        # =====================================================
        # SANITIZE INPUTS
        # =====================================================

        source = str(source).strip().lower()[:100]
        target = str(target).strip().lower()[:100]

        relation = clean_relation(relation)

        repo = normalize_repo(repo)

        # =====================================================
        # VALIDATION
        # =====================================================

        if not source or not target:
            return

        # Prevent tiny garbage nodes
        if len(source) < 2 or len(target) < 2:
            return

        # Prevent self-loops
        if source == target:
            return

        # Prevent invalid repo
        if not repo or repo == "owner/repo":
            return

        # =====================================================
        # CYPHER QUERY
        # =====================================================

        query = f"""
        MERGE (a:Technology {{name: $source}})
        MERGE (b:Technology {{name: $target}})

        MERGE (a)-[r:{relation} {{
            repo: $repo,
            source: $source,
            target: $target
        }}]->(b)

        ON CREATE SET
            r.confidence = $confidence,
            r.created_at = timestamp()

        ON MATCH SET
            r.confidence =
            CASE
                WHEN r.confidence < $confidence
                THEN $confidence
                ELSE r.confidence
            END,
            r.updated_at = timestamp()
        """

        # =====================================================
        # EXECUTE
        # =====================================================

        with driver.session() as session:

            session.run(
                query,
                source=source,
                target=target,
                repo=normalize_repo(repo),
                confidence=confidence,
            )

        print(
            f"✅ Stored repo [{repo}]: "
            f"{source} -[{relation}]-> {target} "
            f"(Confidence: {confidence})"
        )

    except Exception as e:

        print(f"❌ Graph Store Error: {e}")

def fetch_graph(repo: str) -> list[dict]:
    """Fetches all stored relationships for a specific repository from Neo4j."""
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (a)-[r]->(b)
                WHERE r.repo = $repo
                RETURN
                    a.name AS source,
                    type(r) AS relation,
                    b.name AS target
                LIMIT $limit
                """,
                repo=normalize_repo(repo),
                limit=MAX_GRAPH_RELATIONS,
            )

            data = []

            for row in result:
                data.append(
                    {
                        "source": row["source"],
                        "relation": row["relation"],
                        "target": row["target"],
                    }
                )

            print(f"✅ Graph fetched for [{repo}]: {len(data)} relationships")

            return data

    except Exception as e:
        print(f"❌ Graph Fetch Error: {e}")

        return []


# =========================================================
# CODE INTELLIGENCE (MULTILANGUAGE)
# =========================================================


@tool
def analyze_code_structure(code: str, filename: str = "unknown.py") -> str:
    """
    Parse source code to extract structure: imports, classes, functions, and call chains.
    Returns JSON with symbols, useful for building a dependency graph.
    Supports Python, JavaScript, TypeScript, Go, Java, C++, and Rust strictly via tree-sitter.

    Args:
        code: The complete source code as a string.
        filename: Filename or path to infer language, e.g., 'main.py' or 'app.ts'.
    
    Returns:
        JSON object with 'imports', 'classes', 'functions', and 'parser' (tree-sitter).
    """
    ext = filename.split(".")[-1].lower() if "." in filename else "py"
    analysis = {
        "file": filename,
        "language": ext,
        "imports": [],
        "classes": [],
        "functions": [],
        "execution_chains": [],
        "parser": "tree-sitter"
    }

    if get_parser is None:
        raise RuntimeError("Tree-sitter parser is not available. Baseline fallbacks are disabled.")

    lang_map = {
        "py": "python",
        "js": "javascript",
        "jsx": "javascript",
        "ts": "typescript",
        "tsx": "tsx",
        "go": "go",
        "java": "java",
        "cpp": "cpp",
        "rs": "rust"
    }

    if ext not in lang_map:
        raise ValueError(f"Unsupported file extension strictly for tree-sitter parsing: {ext}")

    ts_parser = get_parser(lang_map[ext])
    if ts_parser is None:
        raise RuntimeError(f"Could not load tree-sitter parser for language: {lang_map[ext]}")

    tree = ts_parser.parse(code)

    def walk_ts(node):
        if callable(node):
            node = node()
        if not node:
            return

        type_name = getattr(node, "type", "")
        if callable(type_name):
            type_name = type_name()

        if "function" in type_name or "method" in type_name:
            children = getattr(node, "children", [])
            if callable(children):
                children = children()
            for child in children:
                child_type = getattr(child, "type", "")
                if callable(child_type):
                    child_type = child_type()
                if child_type == "identifier":
                    analysis["functions"].append(
                        code[child.start_byte:child.end_byte]
                    )

        elif "class" in type_name:
            children = getattr(node, "children", [])
            if callable(children):
                children = children()
            for child in children:
                child_type = getattr(child, "type", "")
                if callable(child_type):
                    child_type = child_type()
                if child_type == "identifier":
                    analysis["classes"].append(
                        code[child.start_byte:child.end_byte]
                    )

        elif "import" in type_name or "require" in type_name:
            analysis["imports"].append(
                code[node.start_byte:node.end_byte].strip()
            )

        children = getattr(node, "children", [])
        if callable(children):
            children = children()
        for child in children:
            walk_ts(child)

    root = tree.root_node
    if callable(root):
        root = root()
    walk_ts(root)

    return json.dumps(analysis, indent=2)


# =========================================================
# SECURITY TOOLS (STATIC ANALYSIS)
# =========================================================


@tool
def analyze_security_static(code: str, filename: str = "unknown.py") -> str:
    """
    Perform static security analysis on source code.
    Detects hardcoded secrets, dangerous sinks (eval, exec, pickle), unsafe deserialization, and misconfigurations.
    Returns deterministic findings without speculation.

    Args:
        code: The complete source code as a string.
        filename: Filename or path for reporting context, e.g., 'config.py'.
    
    Returns:
        List of security findings in format: [SEVERITY] risk description.
        Returns "No common vulnerabilities detected" if scan is clean.
    """
    risks = []

    # 1. Dangerous Sinks
    SINKS = {
        r"eval\(": "CRITICAL: Use of eval() detected. Potential Arbitrary Code Execution.",
        r"exec\(": "CRITICAL: Use of exec() detected. Potential Arbitrary Code Execution.",
        r"os\.system\(": "HIGH: OS command injection risk via os.system().",
        r"subprocess\.run\(.*shell=True": "HIGH: Subprocess execution with shell=True is dangerous.",
        r"pickle\.load\(": "HIGH: Unsafe deserialization via pickle.",
        r"yaml\.load\((?!.*Loader=)": "MEDIUM: Potentially unsafe YAML loading.",
    }

    for pattern, message in SINKS.items():
        if re.search(pattern, code):
            risks.append(f"[SINK] {message}")

    # 2. Secret Leakage (Generic Patterns)
    SECRETS = {
        r"(api|secret|token|password|key)\s*=\s*['\"][a-zA-Z0-9]{10,}['\"]": "CRITICAL: Potential hardcoded secret/key detected.",
        r"AWS_SECRET_ACCESS_KEY\s*=": "CRITICAL: Hardcoded AWS Secret detected.",
        r"sk-[a-zA-Z0-9]{20,}": "CRITICAL: Potential OpenAI API Key detected.",
    }

    for pattern, message in SECRETS.items():
        if re.search(pattern, code, re.IGNORECASE):
            risks.append(f"[SECRET] {message}")

    # 3. Network/File Risks
    if "0.0.0.0" in code:
        risks.append("[CONFIG] HIGH: Application binding to 0.0.0.0 (all interfaces).")

    if not risks:
        return f"Static analysis for {filename}: No common vulnerabilities detected."

    return f"Security Audit for {filename}:\n" + "\n".join(risks)


# =========================================================
# GRAPH QUERY TOOL
# =========================================


@tool
async def query_knowledge_graph(repo: str, query_type: str = "all") -> str:
    """
    Query the Neo4j knowledge graph for relationships stored about a repository.
    Returns all Technology nodes and edges (USES, CALLS, DEPENDS_ON, etc.) for the repo.

    Args:
        repo: Repository identifier in 'owner/repo' format, e.g., 'langchain-ai/langgraph'.
             Do NOT use placeholder 'owner/repo'.
        query_type: Query scope (default 'all'). Currently only 'all' is supported.
    
    Returns:
        JSON array of relationships: [{"source": "...", "relation": "...", "target": "..."}].
        Empty array [] if no relationships have been stored yet.
    """
    try:
        data = fetch_graph(repo)
        return truncate_for_api(json.dumps(data, indent=2), max_chars=800)
    except Exception as e:
        return f"GRAPH QUERY ERROR: {str(e)}"


# =========================================================
# MCP MANAGER (CONNECTION POOLING)
# =========================================================

class MCPManager:
    _session: ClientSession = None
    _exit_stack: contextlib.AsyncExitStack = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_session(cls):
        async with cls._lock:
            if cls._session is None:
                cls._exit_stack = contextlib.AsyncExitStack()
                read, write = await cls._exit_stack.enter_async_context(stdio_client(github_server))
                cls._session = await cls._exit_stack.enter_async_context(ClientSession(read, write))
                await cls._session.initialize()
                print("🚀 MCP Persistent Session Initialized")
            return cls._session

    @classmethod
    async def close(cls):
        if cls._exit_stack:
            await cls._exit_stack.aclose()
            cls._session = None
            cls._exit_stack = None
            print("🛑 MCP Session Closed")

# =========================================================
# README TOOL
# =========================================================


@tool
async def github_readme(repo: str) -> str:
    """
    Fetch the README.md file from a GitHub repository to understand project purpose, setup, and key concepts.

    Args:
        repo: Repository identifier in 'owner/repo' format, e.g., 'langchain-ai/langgraph'.
             Do NOT use placeholder 'owner/repo'.
    
    Returns:
        The full README.md content, or an error message if the file is not found.
    """

    try:
        repo = str(repo).strip()
        if not repo or repo == "owner/repo" or repo.count("/") != 1:
            return "ERROR: github_readme requires a concrete repo in 'owner/repo' format, for example 'langchain-ai/langgraph'."

        owner, repo_name = repo.split("/")
        session = await MCPManager.get_session()

        result = await asyncio.wait_for(
            session.call_tool(
                "get_file_contents",
                {"owner": owner, "repo": repo_name, "path": "README.md"},
            ),
            timeout=30
        )

        content_text = ""
        if hasattr(result, "content") and isinstance(result.content, list):
            for item in result.content:
                if hasattr(item, "text"):
                    content_text += item.text

        if not content_text:
            content_text = str(result)

        try:
            add_to_vector_store.invoke({"text": content_text, "source": "README.md", "repo": repo})
        except Exception as e:
            print(f"DEBUG: Auto-index failed for README.md: {e}")

        return content_text[:1200]

    except Exception as e:
        return f"README ERROR: {str(e)}"


# =========================================================
# STRUCTURE TOOL
# =========================================================


@tool
async def github_repo_structure(repo: str) -> str:
    """
    Discover key ecosystem and configuration files in a repository (requirements.txt, package.json, Dockerfile, etc.).

    Args:
        repo: Repository identifier in 'owner/repo' format, e.g., 'langchain-ai/langgraph'.
             Do NOT use placeholder 'owner/repo'.
    
    Returns:
        List of detected files that indicate the tech stack and build environment.
    """

    try:
        if "/" not in repo:
            return "ERROR: Repo must be in 'owner/repo' format."

        repo = normalize_repo(repo)
        parts = repo.split("/")
        owner = parts[0]
        repo_name = parts[1]

        # Prioritize only critical ecosystem and entry-point files to minimize traversal
        candidates = list(PRIORITY_FILES)
        # Add a few common extras but cap at MAX_FILES_ANALYZED
        extras = ["tsconfig.json", "yarn.lock", "pom.xml", "go.mod", "Cargo.toml", ".env"]
        for e in extras:
            if e not in candidates:
                candidates.append(e)

        findings = []
        session = await MCPManager.get_session()

        async def check_file(file_path):
            try:
                result = await asyncio.wait_for(
                    session.call_tool(
                        "get_file_contents",
                        {"owner": owner, "repo": repo_name, "path": file_path},
                    ),
                    timeout=15,
                )
                text = ""
                if hasattr(result, "content") and isinstance(result.content, list):
                    for item in result.content:
                        if hasattr(item, "text"):
                            text += item.text
                if not text:
                    text = str(result)
                
                if text and "error" not in text.lower():
                    return file_path
            except Exception:
                pass
            return None

        # Check all candidates in parallel using asyncio.gather to avoid search_code rate limits
        tasks = [check_file(file) for file in candidates]
        results = await asyncio.gather(*tasks)
        
        for f in results:
            if f:
                findings.append(f"Detected: {f}")

        return "\n".join(findings)

    except Exception as e:
        return f"STRUCTURE ERROR: {str(e)}"


# =========================================================
# GIT TREE TOOL
# =========================================================
@tool
async def github_git_tree(repo: str) -> str:
    """
    Fetch repository tree from GitHub MCP with robust parsing.
    """

    import re

    try:

        repo = normalize_repo(repo)

        if "/" not in repo:
            return "ERROR: Repo must be in owner/repo format"

        owner, repo_name = repo.split("/")

        session = await MCPManager.get_session()

        # Fetch the list of available tools
        available_tools = []
        try:
            tools_list = await session.list_tools()
            available_tools = [t.name for t in tools_list.tools]
            print("\n========== AVAILABLE MCP TOOLS ==========")
            print(available_tools)
            print("=========================================\n")
        except Exception as list_err:
            print("Failed to list tools:", list_err)

        tree_paths = []

        # The github MCP server (@modelcontextprotocol/server-github) does NOT expose
        # get_repository_tree — skip straight to search_code fallback without the
        # useless get_repository_tree attempt that always errors out.
        use_repo_tree = "get_repository_tree" in available_tools  # must be explicit, never default-True

        if use_repo_tree:
            try:
                result = await asyncio.wait_for(
                    session.call_tool(
                        "get_repository_tree",
                        {
                            "owner": owner,
                            "repo": repo_name,
                            "recursive": True,
                        },
                    ),
                    timeout=120,
                )

                print("\n========== RAW MCP RESPONSE ==========\n")

                # =====================================================
                # MCP CONTENT PARSING
                # =====================================================

                if hasattr(result, "content"):

                    for item in result.content:

                        if not hasattr(item, "text"):
                            continue

                        raw = str(item.text)

                        print(raw[:1500])

                        try:

                            parsed = json.loads(raw)

                            print("DEBUG PARSED TYPE:", type(parsed))

                            if isinstance(parsed, dict):
                                print("DEBUG PARSED KEYS:", parsed.keys())

                            # =================================================
                            # CASE 1: {"tree": [...]}
                            # =================================================

                            if isinstance(parsed, dict):

                                # Standard MCP format
                                if "tree" in parsed:

                                    for node in parsed["tree"]:

                                        if isinstance(node, dict):

                                            path = node.get("path")

                                            if path:
                                                tree_paths.append(path)

                                # Alternate MCP format
                                elif "entries" in parsed:

                                    for node in parsed["entries"]:

                                        if isinstance(node, dict):

                                            path = node.get("path")

                                            if path:
                                                tree_paths.append(path)

                                # Another fallback format
                                elif "files" in parsed:

                                    for node in parsed["files"]:

                                        if isinstance(node, dict):

                                            path = node.get("path")

                                            if path:
                                                tree_paths.append(path)

                            # =================================================
                            # CASE 2: direct list
                            # =================================================

                            elif isinstance(parsed, list):

                                for node in parsed:

                                    if isinstance(node, dict):

                                        path = node.get("path")

                                        if path:
                                            tree_paths.append(path)

                        except Exception as parse_error:
                            print(f"DEBUG JSON PARSE FAILED: {parse_error}")
                            raise ValueError(f"Failed to parse tree JSON: {parse_error}") from parse_error
            except Exception as tree_err:
                if "get_directory_contents" not in available_tools:
                    raise tree_err
                print(f"get_repository_tree failed ({tree_err}), falling back to get_directory_contents recursive traversal...")

        if not tree_paths and "get_directory_contents" in available_tools:
            print("Using get_directory_contents recursive traversal fallback...")
            queue = [""]
            visited_dirs = 0
            max_visited_dirs = 30  # Safety limit to avoid rate limits
            
            while queue and visited_dirs < max_visited_dirs:
                current_path = queue.pop(0)
                visited_dirs += 1
                try:
                    dir_result = await asyncio.wait_for(
                        session.call_tool(
                            "get_directory_contents",
                            {
                                "owner": owner,
                                "repo": repo_name,
                                "path": current_path,
                            },
                        ),
                        timeout=30,
                    )
                    
                    if hasattr(dir_result, "content"):
                        for item in dir_result.content:
                            if not hasattr(item, "text"):
                                continue
                            try:
                                parsed = json.loads(item.text)
                                if isinstance(parsed, list):
                                    for node in parsed:
                                        if not isinstance(node, dict):
                                            continue
                                        node_path = node.get("path")
                                        node_type = node.get("type")
                                        if not node_path:
                                            continue
                                        
                                        if should_exclude_path(node_path):
                                            continue
                                            
                                        if node_type == "dir":
                                            if len(node_path.split("/")) <= 4:
                                                queue.append(node_path)
                                        else:
                                            tree_paths.append(node_path)
                                elif isinstance(parsed, dict):
                                    for key, val in parsed.items():
                                        if isinstance(val, dict):
                                            node_path = val.get("path")
                                            node_type = val.get("type")
                                            if node_path and not should_exclude_path(node_path):
                                                if node_type == "dir":
                                                    queue.append(node_path)
                                                else:
                                                    tree_paths.append(node_path)
                            except Exception as parse_error:
                                print(f"DEBUG DIR PARSE FAILED for {current_path}: {parse_error}")
                except Exception as dir_err:
                    print(f"Failed to list directory {current_path}: {dir_err}")

        if not tree_paths and "search_code" in available_tools:
            print("Using search_code query-based file harvesting fallback...")
            search_queries = [
                f"import repo:{owner}/{repo_name}",
                f"def repo:{owner}/{repo_name}",
                f"const repo:{owner}/{repo_name}",
                f"package repo:{owner}/{repo_name}",
            ]
            
            for query in search_queries:
                try:
                    await asyncio.sleep(0.5)  # rate limit safety delay
                    search_result = await asyncio.wait_for(
                        session.call_tool(
                            "search_code",
                            {"q": query},
                        ),
                        timeout=20,
                    )
                    
                    if hasattr(search_result, "content"):
                        for item in search_result.content:
                            if not hasattr(item, "text"):
                                continue
                            
                            try:
                                parsed = json.loads(item.text)
                                items_list = []
                                if isinstance(parsed, dict):
                                    items_list = parsed.get("items", [])
                                elif isinstance(parsed, list):
                                    items_list = parsed
                                    
                                for file_node in items_list:
                                    if isinstance(file_node, dict):
                                        file_path = file_node.get("path")
                                        if file_path:
                                            tree_paths.append(file_path)
                            except Exception as parse_error:
                                print(f"DEBUG SEARCH PARSE FAILED for query '{query}': {parse_error}")
                                matches = re.findall(r'"path"\s*:\s*"([^"]+)"', item.text)
                                tree_paths.extend(matches)
                except Exception as search_err:
                    print(f"Search query '{query}' failed: {search_err}")
            
        # =====================================================
        # CLEAN + FILTER
        # =====================================================

        clean_paths = []

        for path in tree_paths:

            path = str(path).strip()

            if not path:
                continue

            if should_exclude_path(path):
                continue

            depth = len(path.split("/")) - 1

            if depth > MAX_RECURSIVE_DEPTH:
                continue

            clean_paths.append(path)

        # deduplicate
        clean_paths = list(dict.fromkeys(clean_paths))

        print(f"\nDEBUG TREE ITEMS: {len(clean_paths)}")

        if not clean_paths:
            return "ERROR: No valid repository paths extracted"

        # =====================================================
        # PRIORITY SCORING
        # =====================================================

        scored_paths = []

        for path in clean_paths:

            filename = path.lower().split("/")[-1]

            score = 0

            # Critical files
            if filename in [
                "main.py",
                "app.py",
                "server.py",
                "requirements.txt",
                "package.json",
                "pyproject.toml",
                "setup.py",
                "dockerfile",
                "go.mod",
                "cargo.toml",
            ]:
                score += 100

            # Source files
            if any(
                filename.endswith(ext)
                for ext in [
                    ".py",
                    ".js",
                    ".ts",
                    ".tsx",
                    ".jsx",
                    ".go",
                    ".rs",
                    ".java",
                ]
            ):
                score += 50

            # Priority dirs
            if is_priority_path(path):
                score += 40

            # Execution files
            if any(
                keyword in filename
                for keyword in [
                    "main",
                    "app",
                    "server",
                    "agent",
                    "graph",
                    "workflow",
                    "pipeline",
                ]
            ):
                score += 25

            scored_paths.append((score, path))

        # =====================================================
        # SORT
        # =====================================================

        scored_paths.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        compressed_tree = []

        seen = set()

        for _, path in scored_paths:

            if path in seen:
                continue

            seen.add(path)

            compressed_tree.append(path)

            if len(compressed_tree) >= MAX_FILES_ANALYZED:
                break

        # =====================================================
        # DEBUG
        # =====================================================

        print("\n========== PRIORITY FILES ==========\n")

        for p in compressed_tree[:20]:
            print(f"DEBUG FILE: {p}")

        return "\n".join(compressed_tree)

    except Exception as e:

        print(f"TREE ERROR: {e}")

        return f"TREE ERROR: {str(e)}"



@tool
async def github_file_content(
    repo: str,
    path: str,
    start_line: int | str = 1,
    end_line: int | str = 200,
) -> str:
    """
    Fetch important source code/config/docs content from repository.
    Automatically indexes into ChromaDB.
    """

    try:
        if "/" not in repo:
            return "ERROR: Repo must be in 'owner/repo' format."

        repo = normalize_repo(repo)

        owner, repo_name = repo.split("/")

        # =========================================
        # SKIP EXCLUDED PATHS
        # =========================================

        if should_exclude_path(path):
            return f"SKIPPED EXCLUDED PATH: {path}"

        session = await MCPManager.get_session()

        result = await asyncio.wait_for(
            session.call_tool(
                "get_file_contents",
                {
                    "owner": owner,
                    "repo": repo_name,
                    "path": path,
                },
            ),
            timeout=90,
        )

        # =========================================
        # EXTRACT CONTENT
        # =========================================

        content_text = ""

        if hasattr(result, "content") and isinstance(result.content, list):

            for content_item in result.content:

                if hasattr(content_item, "text"):
                    content_text += content_item.text

        if not content_text:
            content_text = str(result)

        if not content_text.strip():
            return f"EMPTY FILE: {path}"

        # =========================================
        # LINE LIMIT PROTECTION
        # =========================================

        lines = content_text.splitlines()

        if len(lines) > MAX_FILE_LINES:
            lines = lines[:MAX_FILE_LINES]

        # Safe int conversion
        try:
            s = max(1, int(start_line))
            e = min(len(lines), int(end_line))
        except Exception:
            s, e = 1, 200

        subset = lines[s - 1 : e]

        final_content = "\n".join(subset)

        # =========================================
        # AUTO VECTOR INDEXING
        # =========================================

        try:
            add_to_vector_store.invoke(
                {
                    "text": final_content,
                    "source": path,
                    "repo": repo,
                }
            )

        except Exception as index_error:
            print(
                f"DEBUG: Vector indexing failed for {path}: {index_error}"
            )

        # =========================================
        # DEBUG
        # =========================================

        print(
            f"DEBUG: Loaded file [{path}] "
            f"({len(final_content)} chars)"
        )

        return truncate_for_api(
            final_content,
            max_chars=4000,
        )

    except Exception as e:
        return f"FILE ERROR: {str(e)}"

# =========================================================
# SCRAPE DOCS TOOL
# =========================================================


@tool
async def scrape_docs(url: str) -> str:
    """
    Extract technical documentation from a URL using ScrapeGraphAI.
    Blocks github.com URLs to prevent accidental GitHub UI scraping (use github_* tools instead).
    Returns structured text extracted from the page, automatically indexed into ChromaDB for RAG.

    Args:
        url: Documentation URL to scrape, e.g., 'https://docs.langchain.com/api/'.
             Must NOT be a github.com URL.

    Returns:
        Extracted technical content as a string, automatically indexed into ChromaDB for RAG.
    """
    if "github.com" in url.lower():
        print(f"DEBUG: Blocking GitHub UI scrape for {url}. Use GitHub MCP instead.")
        return "SCRAPE_SKIP: GitHub URLs are handled via internal API tools."

    print(f"DEBUG: Extracting docs from {url}")

    try:
        res = sgai.extract(
            "Extract all technical content: purpose, installation steps, API usage, "
            "configuration options, dependencies, and key concepts. Be thorough.",
            url=url,
        )

        if res.status != "success":
            print(f"DEBUG: API returned non-success status: {res.status}")
            return f"SCRAPE_ERROR: {res.error or 'Unknown error'}"

        content = str(res.data.json_data) if res.data and res.data.json_data else ""

        print(f"DEBUG: Extracted content length: {len(content)}")

        if len(content.strip()) < 200:
            print(f"DEBUG: Content too short ({len(content.strip())})")
            return "SCRAPE_ERROR: Insufficient content extracted"

        return content

    except Exception as e:
        print(f"DEBUG: Exception in scrape_docs: {str(e)}")
        return f"SCRAPE_ERROR: {str(e)}"



# =========================================================
# GROUNDED KNOWLEDGE EXTRACTION TOOL
# =========================================================


@tool
def store_grounded_relationships(repo: str, relationships: list) -> str:
    """
    Store verified technology relationships into the Neo4j knowledge graph.
    ONLY store relationships extracted from deterministic sources: manifests, AST output, or code.
    Do NOT store inferred or speculative relationships.

    Args:
        repo: Repository identifier in 'owner/repo' format, e.g., 'langchain-ai/langgraph'.
        relationships: Array of relationship objects to store.
                       Each object must have: {"source": "Tech1", "relation": "USES", "target": "Tech2"}
                       Valid relations: USES, CALLS, DEPENDS_ON, IMPLEMENTS, PROVIDES, EXTENDS, IMPORTS, COUPLED_WITH.
    
    Returns:
        Confirmation message with count of stored relationships.
        Error message if format is invalid or source is not trusted.
    """
    try:
        # Accept both a native list (from LLM tool-call) and a JSON-encoded string (legacy callers)
        if isinstance(relationships, str):
            data = json.loads(relationships)
        else:
            data = relationships

        if not isinstance(data, list):
            if isinstance(data, dict) and "relationships" in data:
                data = data["relationships"]
            else:
                return "ERROR: Relationships must be a list of dicts."

        stored_count = 0
        processed_relations = set()  # Prevent duplicates in a single run

        for rel in data:
            source = rel.get("source")
            relation = rel.get("relation")
            target = rel.get("target")

            if source and relation and target:
                key = (
                    str(source).strip().lower(),
                    str(relation).strip().upper(),
                    str(target).strip().lower(),
                )

                if key not in processed_relations:
                    store_relationship(source, relation, target, normalize_repo(repo))
                    processed_relations.add(key)
                    stored_count += 1

        return f"✅ Successfully stored {stored_count} deterministic relationships for {repo}"
    except Exception as e:
        return f"RELATIONSHIP STORAGE ERROR: {str(e)}"


# =========================================================
# KNOWLEDGE BASE UTILITIES
# =========================================================

@tool
def add_to_vector_store(text: str, source: str, repo: str):
    """
    Index technical content into ChromaDB for semantic retrieval during analysis.
    Automatically detects content type and applies appropriate chunking.
    Prevents duplicate embeddings using deterministic IDs.
    """

    JUNK_PHRASES = [
        "skip to content",
        "appearance settings",
        "platform",
        "github settings",
        "unfortunately, i can't",
        "i cannot provide",
        "i am unable to",
    ]

    cleaned = clean_text(text)

    # =========================================================
    # CONTENT TYPE DETECTION
    # =========================================================

    is_manifest = any(
        m in source.lower()
        for m in [
            "requirements.txt",
            "package.json",
            "pyproject.toml",
            "setup.py",
            "go.mod",
            "cargo.toml",
            "pom.xml",
        ]
    )

    is_code = any(
        source.lower().endswith(ext)
        for ext in [
            ".py",
            ".js",
            ".ts",
            ".go",
            ".rs",
            ".c",
            ".cpp",
            ".java",
            ".cs",
        ]
    )

    # =========================================================
    # BASIC VALIDATION
    # =========================================================

    if len(cleaned.strip()) < 50:
        return (
            f"⚠️ Skipped {source}: "
            f"content too short ({len(cleaned.strip())} chars)."
        )

    if not repo or normalize_repo(repo) == "owner/repo":
        return (
            "VECTOR STORE ADD ERROR: "
            "Invalid repository identifier provided."
        )

    try:

        # =========================================================
        # CHUNKING STRATEGY
        # =========================================================

        if is_manifest:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=0,
            )

        elif is_code:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=[
                    "\ndef ",
                    "\nclass ",
                    "\n\n",
                    "\n",
                    " ",
                ],
            )

        else:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                separators=[
                    "\n## ",
                    "\n### ",
                    "\n\n",
                    "\n",
                    " ",
                ],
            )

        chunks = splitter.split_text(cleaned)

        # =========================================================
        # QUALITY FILTERING
        # =========================================================

        clean_chunks = []

        for c in chunks:

            lower = c.lower()

            # Junk filtering
            if any(p in lower for p in JUNK_PHRASES):
                continue

            # Boilerplate filtering
            if any(
                phrase in lower
                for phrase in BOILERPLATE_PHRASES
            ):
                continue

            # Minimum size
            if len(c.strip()) < 60:
                continue

            clean_chunks.append(c)

        # =========================================================
        # LIMIT CHUNKS
        # =========================================================

        clean_chunks = clean_chunks[:MAX_CHUNKS_PER_SOURCE]

        if not clean_chunks:
            return (
                f"⚠️ Skipped {source}: "
                f"No quality content remaining."
            )

        # =========================================================
        # METADATA
        # =========================================================

        normalized_repo = normalize_repo(repo)

        content_type = (
            "manifest"
            if is_manifest
            else "code"
            if is_code
            else "docs"
        )

        metadatas = [
            {
                "source": source,
                "repo": normalized_repo,
                "type": content_type,
            }
            for _ in clean_chunks
        ]

        # =========================================================
        # DETERMINISTIC IDS
        # =========================================================

        chunk_ids = [
            hashlib.md5(
                f"{normalized_repo}:{source}:{chunk}".encode()
            ).hexdigest()
            for chunk in clean_chunks
        ]

        # =========================================================
        # DUPLICATE CHECK
        # =========================================================

        existing = vector_store.get(ids=chunk_ids)

        existing_ids = set(existing.get("ids", []))

        new_chunks = []
        new_metadatas = []
        new_ids = []

        for chunk, meta, cid in zip(
            clean_chunks,
            metadatas,
            chunk_ids,
        ):

            if cid not in existing_ids:
                new_chunks.append(chunk)
                new_metadatas.append(meta)
                new_ids.append(cid)

        if not new_chunks:
            return (
                f"⚠️ Already indexed: {source}"
            )

        # =========================================================
        # VECTOR INSERTION
        # =========================================================

        vector_store.add_texts(
            texts=new_chunks,
            metadatas=new_metadatas,
            ids=new_ids,
        )

        print(
            f"DEBUG: Indexed {len(new_chunks)} "
            f"chunks from {source}"
        )

        return (
            f"✅ Stored {len(new_chunks)} "
            f"clean chunks "
            f"[Type: {content_type}] "
            f"from {source}"
        )

    except Exception as e:

        return (
            f"VECTOR STORE ADD ERROR: {str(e)}"
        )


def delete_repo_embeddings(repo: str) -> str:
    """
    Delete all stored embeddings for a repository from ChromaDB.
    This helps prevent unbounded growth and stale vector index state.
    """
    normalized_repo = normalize_repo(repo)
    if not normalized_repo or normalized_repo == "owner/repo":
        return "DELETE EMBEDDINGS ERROR: Invalid repository identifier provided."

    try:
        vector_store.delete(
            where={"repo": {"$eq": normalized_repo}}
        )
        return f"✅ Deleted embeddings for {repo}"
    except Exception as e:
        return f"DELETE EMBEDDINGS ERROR: {str(e)}"


@tool
def search_vector_store(query: str, repo: str, k: int = MAX_RAG_K):
    """
    Semantic search across indexed code, documentation, and manifests for a repository.
    Returns the most relevant chunks to ground reasoning and prevent hallucination.

    Args:
        query: Semantic query, e.g., 'how is authentication configured' or 'database connection logic'.
        repo: Repository identifier in 'owner/repo' format, e.g., 'langchain-ai/langgraph'.
             Do NOT use placeholder 'owner/repo'.
        k: Number of chunks per content category (code/docs/manifest) to retrieve. Default: 2.
    
    Returns:
        Formatted list of relevant chunks with source, type, and relevance score.
        "No grounded context found" if no matches.
    """
    try:
        final_results = []
        if not repo or normalize_repo(repo) == "owner/repo":
            return "VECTOR STORE SEARCH ERROR: Invalid repository identifier provided."

        normalized_repo = normalize_repo(repo)
        # Enforce policy cap on per-category retrieval
        k = min(k, TOP_K_PER_CATEGORY)

        # PERSPECTIVE 1: Code Structure & Logic
        code_hits = vector_store.similarity_search_with_score(
            query, k=k, filter={
                "repo": normalized_repo,
                "type": "code"
            }
        )

        # PERSPECTIVE 2: Documentation & README
        doc_hits = vector_store.similarity_search_with_score(
            query, k=k, filter={
                "repo": normalized_repo,
                "type": "docs"
            }
        )

        # PERSPECTIVE 3: Dependencies & Environment
        manifest_hits = vector_store.similarity_search_with_score(
            query, k=k, filter={
                "repo": normalized_repo,
                "type": "manifest"
            }
        )

        all_hits = code_hits + doc_hits + manifest_hits

        # Remove duplicates based on content
        seen_content = set()
        for doc, score in all_hits:
            if doc.page_content not in seen_content:
                final_results.append(
                    f"--- CHUNK (Source: {doc.metadata.get('source')}, Type: {doc.metadata.get('type')}) ---\n"
                    f"SCORE: {score:.4f}\n"
                    f"CONTENT:\n{truncate_for_api(doc.page_content, MAX_CHUNK_CHARS)}"
                )
                seen_content.add(doc.page_content)

        return truncate_for_api(
            "\n\n".join(final_results) if final_results else "No grounded context found.",
            max_chars=2000,
        )
    except Exception as e:
        return f"VECTOR STORE SEARCH ERROR: {str(e)}"


# =========================================================
# AGENTS
# =========================================================

# =========================================================

repo_agent = create_agent(
    model=llm,
    tools=[github_readme, github_repo_structure],
    system_prompt=f"""
You are a Repository Intelligence agent. You will be given a repository name.
Use tools to fetch actual data, then summarize factually.

RULES:
- ALWAYS pass the exact repository name (owner/repo) to every tool.
- Extract ONLY: Project Purpose / Core Stack / Entry Points / Critical Dependencies.
- MAX 3 bullets per section. No speculation.
- If a tool returns an error, stop and report that error.
- Do NOT invent data. If you cannot fetch it, say so.

{CONCISE_OUTPUT_RULE}
""",
)

# =========================================================
# ARCHITECTURE AGENT
# =========================================================

architecture_agent = create_agent(
    model=llm,
    tools=[github_repo_structure, github_git_tree, github_file_content, analyze_code_structure],
    system_prompt=f"""
Architecture analyzer. Your task: extract structure only from tool outputs, not from reasoning.

RULES:
- ALWAYS pass the exact repository name (owner/repo) to every tool.
- Use `analyze_code_structure` on core files (main.py, app.py, server.js, etc.).
- Extract ONLY: Directory Layout / Key Modules / Top Call Chains (as tool output, not inferred).
- No architectural speculation. Cite only what tools return.
- MAX 3 bullets per section.

{CONCISE_OUTPUT_RULE}
""",
)

# =========================================================
# DOCS AGENT
# =========================================================

docs_agent = create_agent(
    model=llm,
    tools=[scrape_docs],
    system_prompt="""
You are a Documentation Extraction Specialist.

Rules:
- Blocks all 'github.com' URL scrapes (handled by specialized MCP tools).
- Focused strictly on technical content (installation, API, usage).
- Ignore all UI navigation text.
""",
)

# =========================================================
# INTELLIGENCE AGENT
# =========================================================

intelligence_agent = create_agent(
    model=llm,
    tools=[
        store_grounded_relationships,
        github_file_content,   # for fetching manifests; git-tree removed to prevent duplicate traversal
        search_vector_store,
        analyze_code_structure,
    ],
    system_prompt=f"""
Relationship miner. Extract REAL library/module dependencies from manifest files only.

RULES:
- ALWAYS pass the exact repository name (owner/repo) to every tool.
- The manifest content will be provided directly in your context — read it first.
- Parse every dependency from the manifest into a relationship:
  {{"source": "<repo-name>", "relation": "DEPENDS_ON", "target": "<library-name>"}}
- For internal imports: call analyze_code_structure on 1-2 core .py/.js files.
- Store ONLY real library/package names, NOT vague concepts like 'language model' or 'user prompts'.
- Store via store_grounded_relationships (max 12 per run).
- Output: brief summary of packages stored (3 bullets max).
- NEVER invent or generalize. Only store what the manifest/code literally contains.

{CONCISE_OUTPUT_RULE}
""",
)

# =========================================================
# GRAPH AGENT
# =========================================================

graph_agent = create_agent(
    model=llm,
    tools=[query_knowledge_graph],
    system_prompt=f"""
Graph analyst. Query the knowledge graph and extract insights from the relationships.

RULES:
- ALWAYS pass the exact repository name (owner/repo) to query_knowledge_graph.
- Sections: Graph Insights (3 bullets) / Critical Nodes (3 bullets) / Dependency Risks (3 bullets).
- Only reference relationships returned by query_knowledge_graph. Do NOT speculate on relationships not in the graph.
- A "critical node" is one with high in-degree or out-degree in the graph.
- A "dependency risk" is a relationship that could introduce fragility (e.g., single point of failure).
- If the graph is empty, say so clearly: "No relationships stored yet."

{CONCISE_OUTPUT_RULE}
""",
)

# =========================================================
# SECURITY AGENT
# =========================================================

security_agent = create_agent(
    model=llm,
    tools=[github_file_content, github_repo_structure, analyze_security_static],
    system_prompt=f"""
Security auditor. Use tools to detect and cite real security risks.

RULES:
- ALWAYS pass the exact repository name (owner/repo) to every tool.
- Fetch 2-3 core files (main.py, app.py, server.js, config files, entrypoints) via github_file_content.
- Call analyze_security_static on each file to detect hardcoded secrets, dangerous sinks, or misconfigurations.
- Output ONLY confirmed risks: [CRITICAL|HIGH|MEDIUM] risk description — filename only if available.
- Do NOT speculate on architectural risks or unverified vulnerabilities.
- If no static risks detected, output: "Security Risks: None detected in static analysis."

{CONCISE_OUTPUT_RULE}
""",
)

# =========================================================
# ROADMAP AGENT
# =========================================================

roadmap_agent = create_agent(
    model=llm,
    tools=[github_readme],
    system_prompt=f"""
Onboarding agent. Create a 5-step beginner roadmap based on actual repository content.

RULES:
- ALWAYS pass the exact repository name (owner/repo) to github_readme.
- Fetch the README to understand project purpose and setup.
- Extract the 5 most important beginner steps from the README.
- Format: one line per step, numbered 1-5. No multi-paragraph steps.
- If README is unavailable, base steps on repo_data context provided.
- Do NOT invent steps. Roadmap must match actual repo setup.

{CONCISE_OUTPUT_RULE}
""",
)

# =========================================================
# REPORT AGENT
# =========================================================

report_agent = create_agent(
    model=llm,
    tools=[
        github_readme,
        github_repo_structure,
        github_git_tree,
        github_file_content,
        search_vector_store,
    ],
    system_prompt=f"""
You are a senior technical analyst writing an internal engineering report. Be precise, specific, and factual.

MANDATORY RULES:
- Use ONLY the evidence provided in the user message. Zero invention.
- Every claim must name a specific file, library, module, or relationship from the evidence.
- BANNED phrases (do NOT write these):
    "simple and fast", "easy to use", "5 lines of code", "at scale", "user experience",
    "add more documentation", "natural language understanding model", "user prompts",
    "designed to be", "allows users to", "could introduce fragility", "familiarize yourself"
- Tech Stack: list ACTUAL named libraries (e.g. langchain-groq, neo4j-driver, chromadb, fastapi) — NOT just "Python" or "Docker".
- Architecture: name ACTUAL files and modules from the evidence. Do NOT say "directory structure includes X".
- Relationships: use ONLY relationships from the graph evidence. If the graph has real package edges, list them by name.
- Risks: only cite risks backed by static analysis output or real architectural evidence.
- Roadmap: include actual CLI commands or file paths, not generic advice.
- If a section has no concrete evidence: write exactly "[Section]: No data available."
- Total output: max 900 words.

SECTION FORMAT (use exactly these headers):
## Executive Summary
## Architecture
## Tech Stack
## Dependencies (from graph)
## Security
## Risks
## Top Improvement
## Onboarding Roadmap

{CONCISE_OUTPUT_RULE}
""",
)

# =========================================================
# PLANNER NODE
# =========================================================


async def planner_node(state: RepoState):
    print("PLANNER NODE RUNNING")

    return state


# =========================================================
# PARALLEL NODE  (repository + architecture + docs)
# =========================================================


async def parallel_node(state: RepoState):
    """
    Runs repository, architecture, and docs concurrently.
    Includes caching, deterministic repository traversal,
    and automatic core-file indexing.
    """

    print(f"PARALLEL NODE RUNNING: {state['repo']}")

    # =========================================================
    # INIT
    # =========================================================

    if "errors" not in state:
        state["errors"] = []

    docs_url = str(state.get("docs_url", "")).strip()
    raw_scraped = None
    repo_name = str(state.get("repo", "")).strip()

    # =========================================================
    # REPO / DOC URL VALIDATION
    # =========================================================

    if not repo_name or repo_name == "owner/repo":
        state["errors"].append(
            "ERROR: Invalid repository identifier provided."
        )
        return state

    repo_name_only = repo_name.split("/")[-1].lower()

    if docs_url and (
        repo_name_only not in docs_url.lower()
        and "github.com" not in docs_url.lower()
    ):
        warning_msg = (
            f"⚠️ POTENTIAL MISMATCH: "
            f"Repo '{state['repo']}' vs Docs '{docs_url}'"
        )

        print(warning_msg)

        state["errors"].append(warning_msg)

    # =========================================================
    # CACHE CHECK
    # =========================================================

    async with _URL_LOCKS[docs_url]:

        raw_scraped = get_cached_content(docs_url)

        if raw_scraped:

            print(f"CACHE HIT: {docs_url}")

        else:

            try:
                normalized_repo = normalize_repo(state["repo"])

                existing_data = vector_store.get(
                    where={
                        "$and": [
                            {"source": {"$eq": docs_url}},
                            {"repo": {"$eq": normalized_repo}},
                        ]
                    },
                    limit=1,
                )

                if (
                    existing_data
                    and existing_data["documents"]
                ):
                    print(
                        f"CACHE HIT (Chroma): {docs_url}"
                    )

                    raw_scraped = existing_data["documents"][0]

                    set_cached_content(
                        docs_url,
                        raw_scraped,
                    )

            except Exception as e:
                print(
                    f"DEBUG: Cache lookup failed: {e}"
                )

    # =========================================================
    # PARALLEL AGENTS
    # =========================================================

    repo_coro = repo_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
Repository: {state['repo']}

Extract:
1. Project purpose
2. Core stack
3. Entry points
4. Critical dependencies

Only use tools.
No speculation.
""",
                }
            ]
        }
    )

    arch_coro = architecture_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
Repository: {state['repo']}

Analyze:
1. Directory structure
2. Core modules
3. Imports
4. Call chains
5. Main workflows

Only use evidence from tools.
No assumptions.
""",
                }
            ]
        }
    )

    # =========================================================
    # FETCH DOCS
    # =========================================================

    if docs_url:

        if raw_scraped is None:

            print(f"FETCHING DOCS: {docs_url}")

            try:

                results = await asyncio.gather(
                    repo_coro,
                    arch_coro,
                    scrape_docs.ainvoke({"url": docs_url}),
                )

                repo_result, arch_result, scrape_result = results

                scrape_str = str(scrape_result)

                # =========================================================
                # SCRAPE FAIL
                # =========================================================

                if (
                    not scrape_result
                    or "SCRAPE_ERROR" in scrape_str
                ):
                    err_msg = (
                        f"CRITICAL FAILURE: "
                        f"Scrape failed for {docs_url}"
                    )

                    print(err_msg)

                    state["errors"].append(err_msg)

                    state["docs_data"] = (
                        "No documentation available."
                    )

                    state["repo_data"] = truncate_for_api(
                        repo_result["messages"][-1].content
                    )

                    state["architecture_data"] = (
                        truncate_for_api(
                            arch_result["messages"][-1].content
                        )
                    )

                    return state

                # =========================================================
                # SCRAPE SKIP
                # =========================================================

                if "SCRAPE_SKIP" in scrape_str:

                    print(
                        f"INFO: Scrape skipped "
                        f"for {docs_url}"
                    )

                    state["docs_data"] = (
                        "GitHub handled internally."
                    )

                    state["repo_data"] = truncate_for_api(
                        repo_result["messages"][-1].content
                    )

                    state["architecture_data"] = (
                        truncate_for_api(
                            arch_result["messages"][-1].content
                        )
                    )

                    return state

                raw_scraped = scrape_result

                set_cached_content(
                    docs_url,
                    raw_scraped,
                )

                # =========================================================
                # VECTOR STORE INDEXING
                # =========================================================

                await add_to_vector_store.ainvoke(
                    {
                        "text": raw_scraped,
                        "source": docs_url,
                        "repo": state["repo"],
                    }
                )

            except Exception as e:

                err_msg = (
                    f"CRITICAL FAILURE "
                    f"in parallel_node: {str(e)}"
                )

                print(err_msg)

                state["errors"].append(err_msg)

                state["docs_data"] = str(e)

                return state

        else:
            repo_result, arch_result = await asyncio.gather(
                repo_coro,
                arch_coro,
            )

    else:
        repo_result, arch_result = await asyncio.gather(
            repo_coro,
            arch_coro,
        )
        state["docs_data"] = "No documentation URL provided."

    print(
        "DEBUG: Running deterministic "
        "repository traversal"
    )

    try:
        tree_result = await github_git_tree.ainvoke(
            {"repo": state["repo"]}
        )

        important_paths = []

        if isinstance(tree_result, str):

            for line in tree_result.splitlines():

                line = line.strip()

                if not line:
                    continue

                if "ERROR" in line:
                    continue

                if should_exclude_path(line):
                    continue

                # Directly trust prioritized tree output
                important_paths.append(line)

        # Remove duplicates
        important_paths = list(
            dict.fromkeys(important_paths)
        )

        # Limit fetch count
        important_paths = important_paths[:15]

        print(
            f"DEBUG: Selected "
            f"{len(important_paths)} "
            f"priority files"
        )

        for p in important_paths:
            print(f"DEBUG FILE: {p}")

        # =========================================================
        # AUTO FETCH IMPORTANT FILES
        # =========================================================

        fetch_tasks = []

        for path in important_paths:

            fetch_tasks.append(
                github_file_content.ainvoke(
                    {
                        "repo": state["repo"],
                        "path": path,
                        "start_line": 1,
                        "end_line": 300,
                    }
                )
            )

        await asyncio.gather(*fetch_tasks)

        # =========================================================
        # AUTO CODE STRUCTURE ANALYSIS
        # =========================================================

        analysis_tasks = []

        for path in important_paths:

            # Analyze only source code files
            if not any(
                path.endswith(ext)
                for ext in [
                    ".py",
                    ".js",
                    ".ts",
                    ".tsx",
                    ".go",
                    ".rs",
                    ".java",
                ]
            ):
                continue

            try:

                # Fetch file content again for AST analysis
                file_content = await github_file_content.ainvoke(
                    {
                        "repo": state["repo"],
                        "path": path,
                        "start_line": 1,
                        "end_line": 400,
                    }
                )

                # Skip invalid/error files
                if (
                    not file_content
                    or "FILE ERROR" in str(file_content)
                    or "SKIPPED EXCLUDED PATH" in str(file_content)
                    or "EMPTY FILE" in str(file_content)
                ):
                    continue

                # Queue AST/code analysis
                if len(file_content) < 80:
                    continue
                
                analysis_tasks.append(
                    analyze_code_structure.ainvoke(
                        {
                            "code": file_content,
                            "filename": path,
                        }
                    )
                )

            except Exception as analysis_error:

                print(
                    f"DEBUG: Analysis failed for {path}: "
                    f"{analysis_error}"
                )

        # Run all analyses concurrently
        analysis_results = await asyncio.gather(
            *analysis_tasks,
            return_exceptions=True,
        )

        # Debug output
        successful_analyses = 0

        for result in analysis_results:

            if isinstance(result, Exception):
                print(f"DEBUG: Analysis exception: {result}")
                continue

            successful_analyses += 1

        print(
            f"DEBUG: Completed "
            f"{successful_analyses} "
            f"code structure analyses"
        )

        print(
            "DEBUG: Deterministic "
            "file indexing complete"
        )

    except Exception as traversal_error:
        print(
            f"DEBUG: Traversal failed: "
            f"{traversal_error}"
        )

    # =========================================================
    # FINAL STATE UPDATE

    state["docs_data"] = truncate_for_api(
        raw_scraped
        if raw_scraped
        else "No documentation available.",
        max_chars=3000,
    )

    state["repo_data"] = truncate_for_api(
        repo_result["messages"][-1].content,
        max_chars=2000,
    )

    state["architecture_data"] = truncate_for_api(
        arch_result["messages"][-1].content,
        max_chars=2500,
    )

    return state

# =========================================================
# INTELLIGENCE NODE
# =========================================================


async def intelligence_node(state: RepoState):
    print("INTELLIGENCE NODE RUNNING")

    # If scraping failed and we have no chunks, stop the agent from hallucinating
    if any("Scrape failed" in err for err in state.get("errors", [])):
        print(
            "⚠️ Skipping intelligence extraction due to scrape failure to prevent hallucinations."
        )
        state["intelligence_data"] = (
            "INTELLIGENCE ERROR: Documentation retrieval failed. No grounded extraction possible. Skipping dependency extraction to avoid speculative relationships."
        )
        state["retrieved_chunks"] = []
        return state

    # RAG search — populate retrieved_chunks for sidebar display
    JUNK = [
        "unfortunately",
        "i can't provide",
        "i cannot provide",
        "follow these steps",
        "by following",
        "please visit",
        "access the documentation",
        "i'm unable",
        "i do not have access",
    ]

    # Improved retrieval query (Hybrid Multi-Perspective)
    try:
        # DEBUG CHROMA STATE
        print("DEBUG: Checking ChromaDB Metadata...")
        try:
            peek = vector_store.get(limit=3)
            if peek and peek["metadatas"]:
                print(f"DEBUG: Found {len(peek['metadatas'])} samples. Sample Metadata: {peek['metadatas'][0]}")
            else:
                print("DEBUG: ChromaDB is EMPTY or GET failed.")
        except Exception as e:
            print(f"DEBUG: Chroma Peek failed: {e}")

        # Search for code, docs, and manifests in parallel
        categories = ["code", "docs", "manifest"]
        combined_hits = []
        normalized_repo = state["repo"].strip().lower()

        for cat in categories:
            hits = vector_store.similarity_search_with_score(
                f"technical architecture execution {state['query']}",
                k=MAX_RAG_K,
                filter={
                    "$and": [
                        {"repo": {"$eq": normalized_repo}},
                        {"type": {"$eq": cat}}
                    ]
                }
            )
            print(f"DEBUG: Category [{cat}] found {len(hits)} hits")
            combined_hits.extend(hits)



        state["retrieved_chunks"] = [
            {
                "source": doc.metadata.get("source"),
                "content": truncate_for_api(doc.page_content, MAX_CHUNK_CHARS),
                "score": score,
                "type": doc.metadata.get("type", "unknown")
            }
            for doc, score in combined_hits
            if doc.page_content
            and len(doc.page_content) > 50
            and not any(p in doc.page_content.lower() for p in JUNK)
        ]

        # =========================================================
        # REAL GRAPHRAG: Fused Retrieval with Neighbor Expansion
        # =========================================================

        # 1. Identify seed nodes from the first few vector chunks
        content_sample = "\n".join([c["content"] for c in state["retrieved_chunks"][:3]])
        # Extremely basic extraction of technical terms (capitalized words or backticked)
        potential_nodes = set(re.findall(r"`([\w\-_]+)`", content_sample))
        potential_nodes.update(set(re.findall(r"([A-Z][\w\-_]+)", content_sample)))

        # 2. Expand Context via Graph Traversal
        neighbor_data = []
        if potential_nodes:
            print(f"DEBUG: GraphRAG expanding on {len(potential_nodes)} seed nodes")
            neighbor_data = fetch_neighbors(state["repo"], list(potential_nodes))

        graph_context_expansion = ""
        if neighbor_data:
            graph_context_expansion = "\n\n--- GRAPH NEIGHBORS ---\n"
            for n in neighbor_data[:15]:
                graph_context_expansion += f"{n['origin']} -[{n['relation']}]-> {n['target']}\n"
            graph_context_expansion = truncate_for_api(graph_context_expansion, 600)

        print(
            f"DEBUG: RAG Hybrid retrieved {len(state['retrieved_chunks'])} grounded chunks + {len(neighbor_data)} graph expansions"
        )

    except Exception as e:
        print(f"RAG Background Search Error: {e}")
        state["retrieved_chunks"] = []
        graph_context_expansion = ""

    # Fetch existing Graph Data (GraphRAG Integration)
    existing_graph = fetch_graph(state["repo"])
    graph_context = truncate_for_api(
        json.dumps(existing_graph, indent=2) if existing_graph else "No existing graph data.",
        max_chars=800,
    )

    chunk_summary = truncate_for_api(
        "\n".join(
            f"- {c.get('type')}: {c.get('content', '')[:200]}"
            for c in state.get("retrieved_chunks", [])[:6]
        ),
        max_chars=600,
    )

    # =========================================================
    # PRE-FETCH MANIFESTS — give the LLM concrete evidence
    # so it stores real library names, not vague concepts
    # =========================================================
    manifest_candidates = [
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "package.json",
        "go.mod",
        "Cargo.toml",
        "pom.xml",
    ]
    manifest_content = ""
    for manifest_path in manifest_candidates:
        try:
            raw = await github_file_content.ainvoke({
                "repo": state["repo"],
                "path": manifest_path,
                "start_line": 1,
                "end_line": 150,
            })
            raw_str = str(raw)
            if raw_str and "FILE ERROR" not in raw_str and "EMPTY FILE" not in raw_str and len(raw_str) > 30:
                manifest_content += f"\n\n--- {manifest_path} ---\n{raw_str[:1200]}"
        except Exception:
            pass

    manifest_section = truncate_for_api(
        manifest_content if manifest_content else "No manifest files found.",
        max_chars=1800,
    )

    response = await intelligence_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""Repository: {state['repo']}

Extract REAL package dependencies and store via store_grounded_relationships (max 12 per run).

MANIFEST FILES (primary source — use these first):
{manifest_section}

EXISTING GRAPH (skip already stored):
{graph_context}

RAG CHUNKS (supplementary):
{chunk_summary}

TASK:
1. Read the manifest files above. Extract every package/library listed.
2. For each package: store {{"source": "<repo-short-name>", "relation": "DEPENDS_ON", "target": "<package-name>"}}.
3. ONLY use actual package names from the manifest (e.g. 'langchain', 'fastapi', 'neo4j-driver').
4. NEVER store vague targets like 'language model', 'user prompts', 'NLP', etc.
5. After manifests, optionally call analyze_code_structure on 1 core file for IMPORTS.
6. Skip any relationship already in the existing graph.
7. Output: 3 bullets listing what packages were stored.

DO NOT speculate. Only store what you literally read from the files.
""",
                }
            ]
        }
    )

    state["intelligence_data"] = truncate_for_api(response["messages"][-1].content)

    return state


# =========================================================
# GRAPH NODE
# =========================================================


async def graph_node(state: RepoState):
    print("GRAPH NODE RUNNING")

    # Fetch only relationships belonging to this repository
    graph_data = fetch_graph(state["repo"])

    response = await graph_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
Analyze graph relationships (brief):

{truncate_for_api(json.dumps(graph_data, indent=2), max_chars=800)}
""",
                }
            ]
        }
    )

    state["graph_data"] = truncate_for_api(response["messages"][-1].content)

    return state


# =========================================================
# SECURITY NODE
# =========================================================


async def security_node(state: RepoState):
    print("SECURITY NODE RUNNING")

    response = await security_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""Repository: {state['repo']}

Analyze repository security using tools:

EVIDENCE:
Architecture:
{truncate_for_api(state["architecture_data"], 600)}

Relationships:
{truncate_for_api(state["graph_data"], 600)}

TASK:
1. Fetch 2-3 core files (main.py, app.py, config.py, server.js, etc.) via github_file_content.
2. Run analyze_security_static on each file.
3. Report confirmed findings only: [CRITICAL|HIGH|MEDIUM] risk — filename only if available.
4. If no static risks, output: "Security Risks: None detected."

Do NOT speculate on architectural vulnerabilities. Report confirmed static findings only.
""",
                }
            ]
        }
    )

    state["security_data"] = truncate_for_api(response["messages"][-1].content)

    return state


# =========================================================
# ROADMAP NODE
# =========================================================


async def roadmap_node(state: RepoState):
    print("ROADMAP NODE RUNNING")

    response = await roadmap_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""Repository: {state['repo']}

EVIDENCE:
Project Overview:
{truncate_for_api(state["repo_data"], 500)}

Relationships & Tech Stack:
{truncate_for_api(state["intelligence_data"], 500)}

TASK:
Create a 5-step beginner roadmap. Call github_readme with Repository: {state['repo']} to fetch the README.
Extract the 5 most important setup/learning steps from the README.
Format: one line per step, numbered 1-5.

Do NOT invent steps. Base the roadmap on actual README content.
""",
                }
            ]
        }
    )

    state["roadmap_data"] = truncate_for_api(response["messages"][-1].content, 800)

    return state


# =========================================================
# REPORT NODE
# =========================================================


async def report_node(state: RepoState):
    print("REPORT NODE RUNNING")

    # Pull real graph relationships directly from Neo4j for the report
    # (graph_data is the agent summary, but raw edges are more reliable for tech stack listing)
    raw_graph_edges = fetch_graph(state["repo"])
    raw_edges_text = "\n".join(
        f"  {r['source']} -[{r['relation']}]-> {r['target']}"
        for r in raw_graph_edges[:20]
    ) if raw_graph_edges else "No relationships stored yet."

    # Build richer structured context — keep intelligence_data separate so
    # real package names from manifest parsing are never lost in compression
    try:
        structured_context = {
            "repository": state["repo"],
            "project_overview": truncate_for_api(state["repo_data"], 500),
            "architecture": truncate_for_api(state["architecture_data"], 600),
            "dependencies_from_graph": raw_edges_text,          # raw Neo4j edges — real package names
            "intelligence_summary": truncate_for_api(state["intelligence_data"], 400),  # what was stored
            "graph_analysis": truncate_for_api(state["graph_data"], 400),
            "security": truncate_for_api(state["security_data"], 500),
            "roadmap": truncate_for_api(state["roadmap_data"], 500),
            "errors": state.get("errors", [])[:3],
        }
    except Exception:
        structured_context = {"error": "Failed to structure report context"}

    response = await report_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""Repository: {state['repo']}

Write a precise engineering report using ONLY the evidence below.
Every bullet must name a specific file, module, library, or command from the evidence.

=== EVIDENCE ===

[Project Overview]
{structured_context.get('project_overview', 'N/A')}

[Architecture Modules]
{structured_context.get('architecture', 'N/A')}

[Dependency Graph — raw edges from Neo4j]
{structured_context.get('dependencies_from_graph', 'N/A')}

[Intelligence: Packages Stored]
{structured_context.get('intelligence_summary', 'N/A')}

[Graph Analysis]
{structured_context.get('graph_analysis', 'N/A')}

[Security Findings]
{structured_context.get('security', 'N/A')}

[Onboarding Roadmap]
{structured_context.get('roadmap', 'N/A')}

=== REPORT FORMAT ===
Use the 8 sections below. Max 2 bullets per section.
Every bullet cites its evidence source.
List actual library names from the dependency graph (e.g. langchain-groq, neo4j, chromadb) — never just 'Python'.
Do NOT use any banned phrases.
""",
                }
            ]
        }
    )

    state["final_report"] = truncate_for_api(
        response["messages"][-1].content,
        max_chars=MAX_STATE_FIELD_CHARS * 2,
    )
    return state


# =========================================================
# LANGGRAPH WORKFLOW
# =========================================================

workflow = StateGraph(RepoState)

workflow.add_node("planner", planner_node)
workflow.add_node("parallel", parallel_node)  # repo + arch + docs concurrently
workflow.add_node("intelligence", intelligence_node)
workflow.add_node("graph", graph_node)
workflow.add_node("security", security_node)
workflow.add_node("roadmap", roadmap_node)
workflow.add_node("report", report_node)

# =========================================================
# FLOW
# =========================================================

workflow.set_entry_point("planner")
workflow.add_edge("planner", "parallel")
workflow.add_edge("parallel", "intelligence")
workflow.add_edge("intelligence", "graph")
workflow.add_edge("graph", "security")
workflow.add_edge("security", "roadmap")
workflow.add_edge("roadmap", "report")
workflow.add_edge("report", END)

# =========================================================
# COMPILE
# =========================================================

graph = workflow.compile()

# =========================================================
# MAIN
# =========================================================


async def main():
    try:
        result = await graph.ainvoke(
            {
                "query": "Analyze repository and docs",
                "repo": "langchain-ai/langchain",
                "docs_url": "https://python.langchain.com/docs/",
                "repo_data": "",
                "architecture_data": "",
                "docs_data": "",
                "intelligence_data": "",
                "graph_data": "",
                "security_data": "",
                "roadmap_data": "",
                "retrieved_chunks": [],
                "errors": [],
                "final_report": "",
            }
        )

        print(result["final_report"])
    finally:
        await MCPManager.close()


if __name__ == "__main__":
    asyncio.run(main())
