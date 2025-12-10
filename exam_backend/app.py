from flask import Flask, jsonify, request


from routes.session_routes import start_session, end_session
from routes.event_routes import receive_event
from routes.alert_routes import get_alerts
from models.session_model import Session
from models.event_model import Event
from models.alert_model import Alert
from database.db import Base, engine


app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Backend is running"})

# Session routes (still placeholders)
@app.route("/start_session")
def start_sess():
    return start_session()

@app.route("/end_session")
def end_sess():
    return end_session()

# EVENT route (NOW POST)
@app.route("/event", methods=["POST"])
def event():
    data = request.json
    return receive_event(data)

# Alerts route (placeholder)
@app.route("/alerts")
def alerts():
    return get_alerts()

# Health check
@app.route("/health")
def health():
    return jsonify({"status": "OK", "message": "Backend healthy"})

Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    app.run(debug=True)