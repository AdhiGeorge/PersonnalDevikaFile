import pytest
from src.state import AgentState

@pytest.fixture
def sm():
    return AgentState()

def test_create_state(sm):
    project_name = "Test Project"
    sm.create_state(project_name)
    assert project_name in sm.get_all_states()

def test_add_to_current_state(sm):
    project_name = "Test Project"
    sm.create_state(project_name)
    new_state = sm.new_state()
    new_state["internal_monologue"] = "Working on the first step."
    sm.add_to_current_state(project_name, new_state)
    state_stack = sm.get_current_state(project_name)
    assert len(state_stack) > 0
    assert state_stack[-1]["internal_monologue"] == "Working on the first step."

def test_get_latest_state(sm):
    project_name = "Test Project"
    sm.create_state(project_name)
    new_state = sm.new_state()
    new_state["internal_monologue"] = "Latest state."
    sm.add_to_current_state(project_name, new_state)
    latest = sm.get_latest_state(project_name)
    assert latest["internal_monologue"] == "Latest state."

def test_update_latest_state(sm):
    project_name = "Test Project"
    sm.create_state(project_name)
    new_state = sm.new_state()
    new_state["internal_monologue"] = "Original state."
    sm.add_to_current_state(project_name, new_state)
    updated_state = sm.new_state()
    updated_state["internal_monologue"] = "Updated state."
    sm.update_latest_state(project_name, updated_state)
    latest = sm.get_latest_state(project_name)
    assert latest["internal_monologue"] == "Updated state."

def test_set_agent_completed(sm):
    project_name = "Test Project"
    sm.create_state(project_name)
    sm.set_agent_completed(project_name, True)
    latest = sm.get_latest_state(project_name)
    assert latest["agent_completed"] is True

def test_delete_state(sm):
    project_name = "Test Project"
    sm.create_state(project_name)
    sm.delete_state(project_name)
    assert project_name not in sm.get_all_states() 