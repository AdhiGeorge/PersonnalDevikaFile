import asyncio
import logging
<<<<<<< HEAD
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
from src.agents.agent import Agent
from src.config import Config
from src.apis.project import project_bp, _run_agent
from src.apis.status import status_bp
=======
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from src.agents.agent import Agent
from src.config import Config
from src.logger import Logger, route_logger
from src.apis.project import project_bp
>>>>>>> 925f80e (fifth commit)
from src.socket_instance import socketio, emit_agent
from prometheus_client import start_http_server
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
<<<<<<< HEAD
import threading
from werkzeug.utils import secure_filename
from src.project import ProjectManager
=======
from src.project import ProjectManager
from src.llm import LLM
from src.state import AgentState
from threading import Thread
from src.utils.token_tracker import TokenTracker
import os
>>>>>>> 925f80e (fifth commit)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add console exporter for development
console_exporter = ConsoleSpanExporter()
span_processor = BatchSpanProcessor(console_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Create Flask app
app = Flask(__name__)
<<<<<<< HEAD
# Enable CORS so the Vite dev server (localhost:3000) can call the API
CORS(app, resources={r"/api/*": {"origins": "*"}})
=======
CORS(app)
>>>>>>> 925f80e (fifth commit)
app.register_blueprint(project_bp)
app.register_blueprint(status_bp)

# Initialize Flask-SocketIO
socketio.init_app(app, async_mode='threading')

# Start Prometheus metrics server
config = Config()
prometheus_port = config.get_config()["monitoring"]["metrics"]["prometheus"]["port"]
start_http_server(prometheus_port)

# Initialize agent
agent = Agent(
    base_model=config.get_config()["azure_openai"]["model"],
    search_engine=config.get_config()["search_engines"]["primary"]
)

# Project APIs

@app.route("/api/data", methods=["GET"])
@route_logger(logger)
def data():
    project_manager = ProjectManager()
    project_list = project_manager.get_project_list()
    models = LLM().list_models()
    search_engines = ["Bing", "Google", "DuckDuckGo"]
    return jsonify({"projects": project_list, "models": models, "search_engines": search_engines})


@app.route("/api/messages", methods=["POST"])
@route_logger(logger)
def get_messages():
    data = request.json
    project_name = data.get("project_name")
    messages = ProjectManager().get_messages(project_name)
    return jsonify({"messages": messages})


@app.route("/api/calculate-tokens", methods=["POST"])
@route_logger(logger)
def calculate_tokens():
    data = request.json
    prompt = data.get("prompt")
    model_id = data.get("model_id", "cl100k_base") # Default model if not provided
    tracker = TokenTracker()
    tokens = tracker.count_tokens(prompt, model_id)
    return jsonify({"token_usage": tokens})


@app.route("/api/token-usage", methods=["GET"])
@route_logger(logger)
def token_usage():
    project_name = request.args.get("project_name")
    token_count = AgentState().get_latest_token_usage(project_name)
    return jsonify({"token_usage": token_count})


async def _run_agent_logic(agent_instance, message, project_name):
    try:
        response = await agent_instance.execute(message, project_name)
        logger.info(f"Agent response: {response}")
        agent_instance.make_decision(response, project_name)
    except Exception as e:
        logger.error(f"Error executing agent: {str(e)}")
        emit_agent("info", {"type": "error", "message": f"Error executing agent: {str(e)}"})

def _run_agent_in_thread(agent_instance, message, project_name):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_run_agent_logic(agent_instance, message, project_name))
    loop.close()


# Socket event handlers
@socketio.on('socket_connect')
def handle_socket_connect(data):
    logger.info("Socket connected: %s", data)
    emit_agent("socket_response", {"data": "Server Connected"})

@socketio.on('user-message')
<<<<<<< HEAD
def handle_user_message(data):
    """Handle real-time user prompt via Socket.IO.

    Expected payload from front-end:
    {
        "message": str,
        "base_model": str,
        "project_name": str,
        "search_engine": str
    }
    """
    logger.info("User message received: %s", data)

    try:
        prompt = data.get("message", "")
        if not prompt:
            raise ValueError("Prompt is empty")

        base_model = data.get("base_model") or config.azure_openai.model
        project_name = secure_filename(data.get("project_name", "default"))
        search_engine = data.get("search_engine") or config.search_engines.primary

        # Persist user message to DB / state
        ProjectManager().add_message_from_user(project_name, prompt)

        # Kick off agent execution in background thread (non-blocking)
        threading.Thread(
            target=_run_agent,
            args=(prompt, project_name, base_model, search_engine),
            daemon=True,
        ).start()

    except Exception as e:
        logger.error("Error handling user-message: %s", str(e))
        emit_agent("info", {"type": "error", "message": str(e)})
=======
def handle_message(data):
    logger.info(f"Received data in handle_message: {data}")
    logger.info(f"User message: {data}")
    message = data.get('message')
    base_model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")  # Use deployment name from env var
    project_name = data.get('project_name')
    search_engine = data.get('search_engine').lower()

    agent = Agent(base_model=base_model, search_engine=search_engine)
    agent_state_manager = AgentState()

    state = agent_state_manager.get_latest_state(project_name)
    if not state:
        thread = Thread(target=_run_agent_in_thread, args=(agent, message, project_name))
        thread.start()
    else:
        if agent_state_manager.is_agent_completed(project_name):
            thread = Thread(target=_run_agent_in_thread, args=(agent, message, project_name))
            thread.start()
        else:
            emit_agent("info", {"type": "warning", "message": "previous agent doesn't completed it's task."})
            last_state = agent_state_manager.get_latest_state(project_name)
            if last_state["agent_is_active"] or not last_state["completed"]:
                thread = Thread(target=_run_agent_in_thread, args=(agent, message, project_name))
                thread.start()
            else:
                thread = Thread(target=_run_agent_in_thread, args=(agent, message, project_name))
                thread.start()


@app.route("/api/status", methods=["GET"])
@route_logger(logger)
def status():
    return jsonify({"status": "server is running!"})

@app.route("/api/is-agent-active", methods=["POST"])
@route_logger(logger)
def is_agent_active():
    data = request.json
    project_name = data.get("project_name")
    is_active = AgentState().is_agent_active(project_name)
    return jsonify({"is_active": is_active})

@app.route("/api/get-agent-state", methods=["POST"])
@route_logger(logger)
def get_agent_state():
    data = request.json
    project_name = data.get("project_name")
    agent_state = AgentState().get_latest_state(project_name)
    return jsonify({"state": agent_state})

@app.route("/api/get-browser-snapshot", methods=["GET"])
@route_logger(logger)
def browser_snapshot():
    snapshot_path = request.args.get("snapshot_path")
    return send_file(snapshot_path, as_attachment=True)
>>>>>>> 925f80e (fifth commit)

async def main():
    # Example usage
    project_name = "test_project"
    prompt = "Create a simple Python web server using FastAPI"
    try:
        response = await agent.execute(prompt, project_name)
        logger.info(f"Agent response: {response}")
        agent.make_decision(response, project_name)
    except Exception as e:
        logger.error(f"Error executing agent: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Devika is up and running!")
    # Run the Flask-SocketIO server
    socketio.run(app, debug=False, port=1337, host="0.0.0.0")
