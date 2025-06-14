from flask import blueprints, request, jsonify, send_file, make_response
from werkzeug.utils import secure_filename
from src.logger import Logger, route_logger
from src.config import Config
from src.project import ProjectManager
from ..state import AgentState
from src.agents.agent import Agent
from src.socket_instance import emit_agent
import os
import asyncio
import threading

project_bp = blueprints.Blueprint("project", __name__)

logger = Logger()
manager = ProjectManager()


# Project APIs

@project_bp.route("/api/get-project-files", methods=["GET"])
@route_logger(logger)
def project_files():
    project_name = secure_filename(request.args.get("project_name"))
    files = manager.get_project_files(project_name)  
    return jsonify({"files": files})

@project_bp.route("/api/create-project", methods=["POST"])
@route_logger(logger)
def create_project():
    data = request.json
    project_name = data.get("project_name")
    manager.create_project(secure_filename(project_name))
    return jsonify({"message": "Project created"})


@project_bp.route("/api/delete-project", methods=["POST"])
@route_logger(logger)
def delete_project():
    data = request.json
    project_name = secure_filename(data.get("project_name"))
    manager.delete_project(project_name)
    AgentState().delete_state(project_name)
    return jsonify({"message": "Project deleted"})

# ------------------------------------------------------------------
# Additional endpoints required by the front-end
# ------------------------------------------------------------------

# Utility: run async agent execution in a thread.
def _run_agent(prompt: str, project_name: str, base_model: str, search_engine: str):
    try:
        agent = Agent(base_model=base_model, search_engine=search_engine)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(agent.execute(prompt, project_name))
        # Inform UI via socket
        emit_agent("server-message", {"messages": {"from_agent": True, "message": response}})
        agent.subsequent_execute(response, project_name)
    except Exception as e:
        emit_agent("info", {"type": "error", "message": str(e)})

@project_bp.route("/api/execute-agent", methods=["POST"])
@route_logger(logger)
def execute_agent():
    data = request.json
    prompt = data.get("prompt")
    base_model = data.get("base_model") or Config().azure_openai.model
    project_name = secure_filename(data.get("project_name"))
    search_engine = data.get("search_engine") or Config().search_engines.primary

    # Save user message first
    manager.add_message_from_user(project_name, prompt)

    # Spawn background thread for agent execution
    threading.Thread(target=_run_agent, args=(prompt, project_name, base_model, search_engine), daemon=True).start()

    return jsonify({"message": "Agent execution started"})

@project_bp.route("/api/messages", methods=["POST"])
@route_logger(logger)
def get_messages():
    data = request.json
    project_name = secure_filename(data.get("project_name"))
    msgs = manager.get_messages(project_name) or []
    return jsonify({"messages": msgs})

@project_bp.route("/api/get-agent-state", methods=["POST"])
@route_logger(logger)
def get_agent_state():
    data = request.json
    project_name = secure_filename(data.get("project_name"))
    state = AgentState().get_current_state(project_name) or []
    return jsonify({"state": state})

@project_bp.route("/api/download-project", methods=["GET"])
@route_logger(logger)
def download_project():
    project_name = secure_filename(request.args.get("project_name"))
    manager.project_to_zip(project_name)
    project_path = manager.get_zip_path(project_name)
    return send_file(project_path, as_attachment=False)


@project_bp.route("/api/download-project-pdf", methods=["GET"])
@route_logger(logger)
def download_project_pdf():
    project_name = secure_filename(request.args.get("project_name"))
    pdf_dir = Config().get_pdfs_dir()
    pdf_path = os.path.join(pdf_dir, f"{project_name}.pdf")

    response = make_response(send_file(pdf_path))
    response.headers['Content-Type'] = 'application/pdf'
    return response
