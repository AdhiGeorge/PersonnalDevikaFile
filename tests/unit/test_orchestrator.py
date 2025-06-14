import pytest
from agent.core.orchestrator import AgentOrchestrator

@pytest.fixture
def orchestrator():
    return AgentOrchestrator()

def test_run_orchestrator(orchestrator):
    messages = [
        {"role": "user", "content": "Build a FastAPI web server that returns 'Hello, World!' on the root endpoint."}
    ]
    context_variables = {
        "research_needed": True,
        "code_generation_needed": True,
        "execution_needed": True,
        "errors_found": False,
        "task_completed": False
    }
    result = orchestrator.run(
        start_agent="planner",
        messages=messages,
        context_variables=context_variables,
        max_turns=6
    )
    assert result["state"] in ["completed", "error"]
    assert "turn_count" in result["metadata"]
    assert len(result["messages"]) > 0
    assert "context_variables" in result

def test_run_orchestrator_with_invalid_agent(orchestrator):
    messages = [
        {"role": "user", "content": "Build a FastAPI web server."}
    ]
    context_variables = {
        "research_needed": True,
        "code_generation_needed": True,
        "execution_needed": True,
        "errors_found": False,
        "task_completed": False
    }
    result = orchestrator.run(
        start_agent="invalid_agent",
        messages=messages,
        context_variables=context_variables,
        max_turns=6
    )
    assert result["state"] == "error"
    assert "error" in result["messages"][-1]["content"].lower()

def test_run_orchestrator_with_max_turns_exceeded(orchestrator):
    messages = [
        {"role": "user", "content": "Build a FastAPI web server."}
    ]
    context_variables = {
        "research_needed": True,
        "code_generation_needed": True,
        "execution_needed": True,
        "errors_found": False,
        "task_completed": False
    }
    result = orchestrator.run(
        start_agent="planner",
        messages=messages,
        context_variables=context_variables,
        max_turns=1
    )
    assert result["state"] == "error"
    assert "max turns exceeded" in result["messages"][-1]["content"].lower() 