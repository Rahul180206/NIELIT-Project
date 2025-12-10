def parse_event(data):
    # Basic validation
    if "session_id" not in data:
        return {"status": "error", "message": "session_id missing"}

    if "type" not in data:
        return {"status": "error", "message": "event type missing"}

    return {
        "status": "ok",
        "incoming_event": data
    }
