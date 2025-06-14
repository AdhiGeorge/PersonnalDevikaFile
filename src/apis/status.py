from flask import Blueprint, jsonify

from src.project import ProjectManager
from src.state import AgentState
from src.config import Config

# ----------------------------------------------------------------------
# Basic health & data endpoints used by the front-end for polling.
# ----------------------------------------------------------------------

status_bp = Blueprint("status_bp", __name__)


@status_bp.route("/api/status", methods=["GET"])
def get_status():
    """Simple liveness check so UI can verify backend is up."""
    return jsonify({"ok": True})


@status_bp.route("/api/data", methods=["GET"])
def get_data():
    """Return list of projects and whether each agent is currently active."""
    pm = ProjectManager()
    projects = pm.get_project_list()

    state_mgr = AgentState()
    active = {
        name: bool(state_mgr.is_agent_active(name)) or False for name in projects
    }

    # Provide extra data required by the front-end (models, search engines)
    cfg = Config()

    # Provide models in structure expected by UI: { provider: [[id, friendly_name]] }
    models_dict = {}

    # Azure OpenAI (default)
    azure_model = cfg.azure_openai.model
    if azure_model:
        models_dict.setdefault("Azure", []).append([azure_model, "Azure OpenAI"])

    # Extra configured models (generic list)
    extra = cfg.get("llm.models", [])
    if isinstance(extra, list):
        for mid in extra:
            models_dict.setdefault("Custom", []).append([mid, mid])

    # Build search engines list (deduplicated)
    search_engines = [cfg.search_engines.primary]
    extra_eng = cfg.get("search_engines.available", [])
    if isinstance(extra_eng, list):
        search_engines.extend(extra_eng)

    return jsonify({
        "projects": projects,
        "active": active,
        "models": models_dict,
        "search_engines": list(dict.fromkeys(search_engines)),
    })
