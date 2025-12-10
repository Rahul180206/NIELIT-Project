# backend/app.py
# Main Flask backend for:
# - receiving events from proctor engine (/api/events)
# - providing summary, raw logs, and timeline for dashboard.

from flask import Flask, request, jsonify
from datetime import datetime
import os
import json

from aggregator import get_summary, get_raw_events, get_timeline

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "backend_logs")
os.makedirs(LOG_DIR, exist_ok=True)


@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify(
        {
            "status": "ok",
            "time": datetime.utcnow().isoformat() + "Z",
        }
    )


@app.route("/api/events", methods=["POST"])
def events():
    """
    Endpoint called by your EventSender from main_proctoring_v1_1.py.
    Accepts either:
      - a single JSON object
      - a JSON array of objects
    Saves them into backend_logs/events_YYYYMMDD.jsonl
    """
    try:
        data = request.get_json(force=True)

        # Accept both single object and list
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Expected JSON list or object",
                    }
                ),
                400,
            )

        # Append to daily backend event log
        fname = os.path.join(
            LOG_DIR, f"events_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        )
        with open(fname, "a") as f:
            for ev in data:
                f.write(json.dumps(ev) + "\n")

        print(f"[BACKEND] Received {len(data)} events")
        return jsonify({"status": "ok", "received": len(data)})

    except Exception as e:
        print("[BACKEND] Error in /api/events:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# ======================================================
# NEW: Dashboard / analytics APIs (for Member 5)
# ======================================================

@app.route("/api/session/summary", methods=["GET"])
def session_summary():
    """
    Returns summary for a given date (or latest if no ?date=).
    Example:
        /api/session/summary          -> latest
        /api/session/summary?date=20251208
    """
    date_str = request.args.get("date")  # YYYYMMDD or None
    summary = get_summary(date_str=date_str)
    if summary is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "No event log found for given date",
                }
            ),
            404,
        )

    return jsonify({"status": "ok", "data": summary})


@app.route("/api/session/raw", methods=["GET"])
def session_raw():
    """
    Returns raw events (limited) for a given date.
    Params:
        date: YYYYMMDD or omitted for latest
        limit: max number of events (default 1000)
    """
    date_str = request.args.get("date")
    limit_str = request.args.get("limit")
    limit = None
    if limit_str:
        try:
            limit = int(limit_str)
        except ValueError:
            limit = 1000

    raw = get_raw_events(date_str=date_str, limit=limit or 1000)
    if raw is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "No event log found for given date",
                }
            ),
            404,
        )

    return jsonify({"status": "ok", "data": raw})


@app.route("/api/session/timeline", methods=["GET"])
def session_timeline():
    """
    Returns time-ordered events for plotting charts.
    Params:
        date: YYYYMMDD or omitted for latest
    """
    date_str = request.args.get("date")
    tl = get_timeline(date_str=date_str)
    if tl is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "No event log found for given date",
                }
            ),
            404,
        )

    return jsonify({"status": "ok", "data": tl})


if __name__ == "__main__":
    # Accessible at http://127.0.0.1:5000
    app.run(host="127.0.0.1", port=5000, debug=True)
