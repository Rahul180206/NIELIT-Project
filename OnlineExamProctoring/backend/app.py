from flask import Flask, request, jsonify
from datetime import datetime
import os
import json

from rules_engine import RuleEngine
from aggregator import EventAggregator

app = Flask(__name__)

LOG_DIR = "backend_logs"
os.makedirs(LOG_DIR, exist_ok=True)

engine = RuleEngine()
aggregator = EventAggregator()

@app.route("/api/ping")
def ping():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"})

@app.route("/api/events", methods=["POST"])
def events():
    try:
        data = request.get_json(force=True)

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400

        filename = os.path.join(LOG_DIR, f"events_{datetime.utcnow().strftime('%Y%m%d')}.jsonl")

        with open(filename, "a") as f:
            for ev in data:
                f.write(json.dumps(ev) + "\n")

                # Apply rule engine
                alert = engine.process_event(ev)

                # Update analytics
                aggregator.update(ev, alert)

        print("[BACKEND] Received", len(data), "events")

        return jsonify({"status": "ok", "received": len(data)})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/session-summary", methods=["GET"])
def summary():
    return jsonify(aggregator.get_summary())

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
