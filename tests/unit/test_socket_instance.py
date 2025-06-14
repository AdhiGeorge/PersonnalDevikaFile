import pytest
from src.socket_instance import emit_agent

def test_emit_agent():
    # This test will only work if a Flask app and SocketIO server are running
    result = emit_agent("test-channel", {"msg": "Hello from socket_instance test!"}, log=True)
    assert result is not None 