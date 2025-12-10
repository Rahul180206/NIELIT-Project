def evaluate_event(event):
    violations = []

    # 1. PHONE RULES
    if event.get("type") == "object_detection":
        if event.get("phone") is True:
            violations.append({"alert_type": "phone_detected", "deduction": 3})

        if event.get("persons", 1) > 1:
            violations.append({"alert_type": "multiple_persons", "deduction": 4})

    # 2. EYE / MOUTH RULES
    if event.get("type") == "eye_mouth":
        if event.get("ear") is not None and event["ear"] < 0.20:
            violations.append({"alert_type": "eye_closed", "deduction": 1})

        if event.get("mar") is not None and event["mar"] > 0.65:
            violations.append({"alert_type": "talking_detected", "deduction": 1})

        if event.get("gaze") in ["left", "right"]:
            violations.append({"alert_type": "looking_away", "deduction": 0.5})

    # 3. HEAD POSE RULES
    if event.get("type") == "head_pose":
        if event.get("yaw", 0) > 30:
            violations.append({"alert_type": "head_turned", "deduction": 2})

        if event.get("pitch", 0) > 20:
            violations.append({"alert_type": "looking_down", "deduction": 1})

    return violations
