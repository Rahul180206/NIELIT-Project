class EventAggregator:
    def __init__(self):
        self.counts = {
            "eye_closure": 0,
            "talking": 0,
            "gaze_left": 0,
            "gaze_right": 0,
            "gaze_up": 0,
            "away": 0
        }

    def update(self, ev, alert):
        et = ev.get("type")

        if et == "LONG_EYE_CLOSURE":
            self.counts["eye_closure"] += 1
        elif et == "TALKING":
            self.counts["talking"] += 1
        elif et == "GAZE_LEFT":
            self.counts["gaze_left"] += 1
        elif et == "GAZE_RIGHT":
            self.counts["gaze_right"] += 1
        elif et == "GAZE_UP":
            self.counts["gaze_up"] += 1
        elif et == "AWAY":
            self.counts["away"] += 1

    def get_summary(self):
        return self.counts
