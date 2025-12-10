from database.db import SessionLocal
from models.session_model import Session
from datetime import datetime
import uuid

def start_session():
    db = SessionLocal()

    # Generate a unique session ID
    session_id = "S" + uuid.uuid4().hex[:6].upper()

    new_session = Session(
        session_id=session_id,
        user_id="user1",   # later dynamic
        start_time=str(datetime.now()),
        end_time=None,
        cheating_score=10, # fair play score starts at 10
        status="active"
    )

    db.add(new_session)
    db.commit()

    return {
        "message": "Session started successfully",
        "session_id": session_id,
        "cheating_score": 10,
        "status": "active"
    }


def end_session():
    return {"message": "Session ended (placeholder)"}
