from database.db import SessionLocal
from models.event_model import Event
from models.alert_model import Alert
from models.session_model import Session
from utils.event_parser import parse_event
from utils.rules_engine import evaluate_event
from datetime import datetime

def receive_event(data):
    db = SessionLocal()

    # Save raw event first
    new_event = Event(
        session_id=data.get("session_id"),
        event_type=data.get("type"),
        raw_data=str(data),
        timestamp=str(datetime.now())
    )
    db.add(new_event)
    db.commit()

    # Apply rules
    violations = evaluate_event(data)

    # Calculate total deduction for this event
    total_deduction = sum(v["deduction"] for v in violations)

    # Update session score
    session = db.query(Session).filter(Session.session_id == data.get("session_id")).first()

    if session:
        new_score = max(0, session.cheating_score - total_deduction)
        session.cheating_score = new_score
        db.commit()

    # Save alerts
    for v in violations:
        alert = Alert(
            session_id=data.get("session_id"),
            alert_type=v["alert_type"],
            severity=v["deduction"],  # store deduction
            message=f"{v['alert_type']} detected",
            timestamp=str(datetime.now())
        )
        db.add(alert)

    db.commit()

    return {
        "message": "Event saved",
        "deduction": total_deduction,
        "remaining_score": session.cheating_score,
        "alerts": violations
    }
