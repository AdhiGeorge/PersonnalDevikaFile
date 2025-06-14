import pytest
from src.project import ProjectManager

@pytest.fixture
def pm():
    return ProjectManager()

def test_create_project(pm):
    project_name = "Test Project"
    pm.create_project(project_name)
    assert project_name in pm.get_all_projects()

def test_add_message_from_user(pm):
    project_name = "Test Project"
    pm.create_project(project_name)
    pm.add_message_from_user(project_name, "Hello, this is a test message.")
    messages = pm.get_messages(project_name)
    assert len(messages) > 0
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Hello, this is a test message."

def test_add_message_from_agent(pm):
    project_name = "Test Project"
    pm.create_project(project_name)
    pm.add_message_from_agent(project_name, "I am the agent, responding to your message.")
    messages = pm.get_messages(project_name)
    assert len(messages) > 0
    assert messages[-1]["role"] == "agent"
    assert messages[-1]["content"] == "I am the agent, responding to your message."

def test_get_all_messages_formatted(pm):
    project_name = "Test Project"
    pm.create_project(project_name)
    pm.add_message_from_user(project_name, "User message.")
    pm.add_message_from_agent(project_name, "Agent message.")
    formatted = pm.get_all_messages_formatted(project_name)
    assert len(formatted) > 0
    assert "User message." in formatted
    assert "Agent message." in formatted

def test_get_project_files(pm):
    project_name = "Test Project"
    pm.create_project(project_name)
    files = pm.get_project_files(project_name)
    assert isinstance(files, list)

def test_delete_project(pm):
    project_name = "Test Project"
    pm.create_project(project_name)
    pm.delete_project(project_name)
    assert project_name not in pm.get_all_projects() 