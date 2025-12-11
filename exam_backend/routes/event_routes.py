from database.db import SessionLocal
from models.event_model import Event
from models.alert_model import Alert
from models.session_model import Session
from utils.event_parser import parse_event
from utils.rules_engine import evaluate_event
from datetime import datetime
import os, json

# JSONL LOG DIRECTORY (same as Member 2)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "backend_logs")
os.makedirs(LOG_DIR, exist_ok=True)


def receive_event(data):
    db = SessionLocal()

    new_event = Event(
        session_id=data.get("session_id"),
        event_type=data.get("type"),
        raw_data=str(data),
        timestamp=str(datetime.now())
    )
    db.add(new_event)
    db.commit()

    violations = evaluate_event(data)
    total_deduction = sum(v["deduction"] for v in violations)

    session = db.query(Session).filter(Session.session_id == data.get("session_id")).first()

    if session:
        new_score = max(0, session.cheating_score - total_deduction)
        session.cheating_score = new_score
        db.commit()
    else:
        new_score = None  # no session found

  
    for v in violations:
        alert = Alert(
            session_id=data.get("session_id"),
            alert_type=v["alert_type"],
            severity=v["deduction"],
            message=f"{v['alert_type']} detected",
            timestamp=str(datetime.now())
        )
        db.add(alert)
    db.commit()

   
    jsonl_filename = os.path.join(
        LOG_DIR,
        f"events_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    )

    with open(jsonl_filename, "a") as f:
        f.write(json.dumps(data) + "\n")

    return {
        "message": "Event saved",
        "deduction": total_deduction,
        "remaining_score": new_score,
        "alerts": violations,
        "jsonl_log": jsonl_filename
    }
