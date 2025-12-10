# backend/rules_engine.py
# Simple rule engine to convert raw events into:
# - counts per type
# - suspicious score (0–100)
# - human-readable verdict and flags

from collections import Counter


def analyze_events(events):
    """
    Input:
        events: list of dicts, each like:
            {
                "type": "TALKING" | "GAZE_LEFT" | ...,
                "frame": 123,
                "ts": 17648xxxxx,
                ...
            }

    Output:
        dict with:
            - total_events
            - counts (per event type)
            - suspicious_score (0–100)
            - verdict ("Normal", "Watch", "Suspicious")
            - flags (extra info)
    """
    result = {
        "total_events": 0,
        "counts": {},
        "suspicious_score": 0,
        "verdict": "Normal",
        "flags": [],
    }

    if not events:
        return result

    # ------------------------------------------------------------------
    # 1) Basic counts
    # ------------------------------------------------------------------
    types = [ev.get("type", "UNKNOWN") for ev in events]
    counts = Counter(types)

    talking_count = counts.get("TALKING", 0)
    gaze_left = counts.get("GAZE_LEFT", 0)
    gaze_right = counts.get("GAZE_RIGHT", 0)
    gaze_up = counts.get("GAZE_UP", 0)
    long_eye = counts.get("LONG_EYE_CLOSURE", 0)
    away_count = counts.get("AWAY", 0)

    total_events = len(events)

    result["total_events"] = total_events
    result["counts"] = dict(counts)

    # ------------------------------------------------------------------
    # 2) Score model (very simple, but good for report)
    # ------------------------------------------------------------------
    score = 0

    # Talking: frequent talking increases suspicion
    # each talking event = +2 points (capped at 30)
    score += min(talking_count * 2, 30)

    # Gaze deviations (left/right/up)
    gaze_total = gaze_left + gaze_right + gaze_up
    # each gaze deviation = +1.5 points (capped at 30)
    score += min(int(gaze_total * 1.5), 30)

    # Long eye closure: stronger evidence
    # each long closure = +4 points
    score += min(long_eye * 4, 20)

    # AWAY (face missing): very strong
    # each away = +5 points
    score += min(away_count * 5, 20)

    # clamp to 0–100
    score = max(0, min(score, 100))
    result["suspicious_score"] = score

    # ------------------------------------------------------------------
    # 3) Verdict based on score
    # ------------------------------------------------------------------
    if score < 30:
        verdict = "Normal"
    elif score < 60:
        verdict = "Watch"
    else:
        verdict = "Suspicious"

    result["verdict"] = verdict

    # ------------------------------------------------------------------
    # 4) Flags for extra explanation
    # ------------------------------------------------------------------
    flags = []

    if talking_count >= 20:
        flags.append("Frequent talking detected")

    if gaze_total >= 20:
        flags.append("Frequent gaze deviations (LEFT/RIGHT/UP)")

    if long_eye >= 5:
        flags.append("Multiple long eye closures")

    if away_count >= 3:
        flags.append("Face missing multiple times (AWAY)")

    # Additional gentle flags
    if 0 < talking_count < 20:
        flags.append(f"Talking events: {talking_count}")
    if gaze_total > 0 and gaze_total < 20:
        flags.append(f"Gaze deviation events: {gaze_total}")
    if long_eye > 0 and long_eye < 5:
        flags.append(f"Long eye closure events: {long_eye}")
    if away_count > 0 and away_count < 3:
        flags.append(f"AWAY events: {away_count}")

    result["flags"] = flags

    return result
