# Analysis policy for repository intelligence

PRIORITY_FILES = [
    # Python
    "main.py",
    "app.py",
    "server.py",
    "run.py",
    "manage.py",
    "config.py",
    "settings.py",

    # Dependency / Build
    "requirements.txt",
    "pyproject.toml",
    "poetry.lock",
    "Dockerfile",
    "docker-compose.yml",

    # JS / TS
    "package.json",
    "tsconfig.json",
    "vite.config.ts",
    "next.config.js",

    # Rust / Go / Java
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",

    # Docs
    "README.md",
]

PRIORITY_DIRS = [
    "api",
    "src",
    "core",
    "graph",
    "graphs",
    "agent",
    "agents",
    "services",
    "service",
    "routers",
    "routes",
    "controllers",
    "models",
    "workflow",
    "workflows",
    "pipeline",
    "pipelines",
    "orchestrator",
    "config",
    "backend",
    "database",
    "db",
]

# Files/dirs to ignore entirely
EXCLUDE_DIRS = {
    # JS
    "node_modules",
    ".next",

    # Python
    "venv",
    ".venv",
    "__pycache__",

    # Build
    "dist",
    "build",
    "coverage",
    ".coverage",

    # Git
    ".git",
    ".github",

    # Cache
    ".mypy_cache",
    ".pytest_cache",

    # Assets
    "assets",
    "images",
    "static",
    "media",

    # Misc
    "logs",
    "tmp",
    "temp",
    "notebooks",
    ".idea",
    ".vscode",
}

# Traversal limits
MAX_RECURSIVE_DEPTH = 4
MAX_FILES_ANALYZED = 40
MAX_FILE_LINES = 2000

# RAG / retrieval settings
TOP_K_PER_CATEGORY = 5
MAX_RAG_K = 5

# Chunk storage caps
MAX_CHUNKS_PER_SOURCE = 30

# Content filters (common tutorial/boilerplate phrases to remove)
BOILERPLATE_PHRASES = [
    "getting started",
    "tutorial",
    "example",
    "quick start",
    "hello world",
    "this repo is for",
    "used for demonstration",
    "follow these steps",
    "by following these steps",
    "unfortunately",
    "i can't provide",
    "i cannot provide",
    "please visit",
    "access the documentation",
]


def is_priority_path(path: str) -> bool:
    """Return True if path is in priority files or directories."""
    p = path.strip().lower()

    for f in PRIORITY_FILES:
        if p.endswith(f):
            return True

    for d in PRIORITY_DIRS:
        if (
            f"/{d}/" in p
            or p.startswith(d + "/")
            or p.endswith("/" + d)
        ):
            return True

    return False


def should_exclude_path(path: str) -> bool:
    p = path.strip().lower()

    for ex in EXCLUDE_DIRS:
        if (
            f"/{ex}/" in p
            or p.startswith(ex + "/")
            or p.endswith("/" + ex)
        ):
            return True

    return False