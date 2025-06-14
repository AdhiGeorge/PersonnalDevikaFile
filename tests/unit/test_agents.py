import pytest
from agent.core.agents import AgentManager

@pytest.fixture
def manager():
    return AgentManager()

def test_run_planner_agent(manager):
    messages = [
        {"role": "user", "content": "Create a project plan for a REST API that manages a todo list."}
    ]
    context_variables = {
        "research_needed": True,
        "code_generation_needed": True,
        "execution_needed": False,
        "errors_found": False,
        "task_completed": False
    }
    result = manager.run(
        agent_name="planner",
        messages=messages,
        context_variables=context_variables
    )
    assert result["agent"] == "planner"
    assert len(result["messages"]) > 0
    assert "context_variables" in result

def test_run_researcher_agent(manager):
    messages = [
        {"role": "user", "content": "Research best practices for building a REST API."}
    ]
    context_variables = {
        "research_needed": True,
        "code_generation_needed": False,
        "execution_needed": False,
        "errors_found": False,
        "task_completed": False
    }
    result = manager.run(
        agent_name="researcher",
        messages=messages,
        context_variables=context_variables
    )
    assert result["agent"] == "researcher"
    assert len(result["messages"]) > 0
    assert "context_variables" in result

def test_run_coder_agent(manager):
    messages = [
        {"role": "user", "content": "Write a FastAPI endpoint that returns a list of todos."}
    ]
    context_variables = {
        "research_needed": False,
        "code_generation_needed": True,
        "execution_needed": False,
        "errors_found": False,
        "task_completed": False
    }
    result = manager.run(
        agent_name="coder",
        messages=messages,
        context_variables=context_variables
    )
    assert result["agent"] == "coder"
    assert len(result["messages"]) > 0
    assert "context_variables" in result

def test_run_executor_agent(manager):
    messages = [
        {"role": "user", "content": "Run the FastAPI server."}
    ]
    context_variables = {
        "research_needed": False,
        "code_generation_needed": False,
        "execution_needed": True,
        "errors_found": False,
        "task_completed": False
    }
    result = manager.run(
        agent_name="executor",
        messages=messages,
        context_variables=context_variables
    )
    assert result["agent"] == "executor"
    assert len(result["messages"]) > 0
    assert "context_variables" in result

def test_run_agent_with_invalid_name(manager):
    messages = [
        {"role": "user", "content": "Do something."}
    ]
    context_variables = {
        "research_needed": True,
        "code_generation_needed": True,
        "execution_needed": True,
        "errors_found": False,
        "task_completed": False
    }
    result = manager.run(
        agent_name="invalid_agent",
        messages=messages,
        context_variables=context_variables
    )
    assert result["agent"] == "invalid_agent"
    assert "error" in result["messages"][-1]["content"].lower() 