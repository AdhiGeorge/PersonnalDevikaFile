# socketio_instance.py
from flask_socketio import SocketIO
from src.logger import Logger
socketio = SocketIO(cors_allowed_origins="*", async_mode="threading")

logger = Logger()


def emit_agent(channel, content, log=True):
    try:
        socketio.emit(channel, content)
        if log:
            logger.info(f"SOCKET {channel} MESSAGE: {content}")
        return True
    except Exception as e:
        logger.error(f"SOCKET {channel} ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    # Real, practical example usage of the socketio emit_agent function
    try:
        # This will only work if a Flask app and SocketIO server are running
        print("Emitting a test message on 'test-channel'...")
        result = emit_agent("test-channel", {"msg": "Hello from socket_instance example!"}, log=True)
        print(f"Emit result: {result}")
    except Exception as e:
        print(f"Error in socket_instance example: {str(e)}")
