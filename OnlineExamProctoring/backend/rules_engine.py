import json
from datetime import datetime

# RULE ENGINE FOR CHEATING / ALERT LOGIC

class RuleEngine:
    def __init__(self):
        self.alerts = []

    def process_event(self, ev):
        et = ev.get("type")

        # Basic rules
        if et == "LONG_EYE_CLOSURE":
            return "ALERT_EYE_CLOSED"

        if et == "TALKING":
            return "ALERT_TALKING"

        if et in ("GAZE_LEFT", "GAZE_RIGHT", "GAZE_UP"):
            return f"ALERT_{et}"

        if et == "AWAY":
            return "ALERT_AWAY"

        return None

