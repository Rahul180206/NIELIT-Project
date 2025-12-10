# backend/aggregator.py
# Utilities to read JSONL event logs and build:
# - raw event list
# - timeline data
# - summary using the rule engine

import os
import json
import glob
from datetime import datetime

from rules_engine import analyze_events

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "backend_logs")


def _list_log_files():
    """
    Returns sorted list of backend_logs/events_YYYYMMDD.jsonl files.
    """
    pattern = os.path.join(LOG_DIR, "events_*.jsonl")
    files = glob.glob(pattern)
    files.sort()  # lexicographic → works for YYYYMMDD
    return files


def _choose_file_for_date(date_str=None):
    """
    If date_str given (format: 'YYYYMMDD'), choose that file.
    Else choose the latest events_*.jsonl file.
    """
    if date_str:
        fname = os.path.join(LOG_DIR, f"events_{date_str}.jsonl")
        if os.path.exists(fname):
            return fname
        return None

    # No date: choose latest file
    files = _list_log_files()
    if not files:
        return None
    return files[-1]


def load_events(date_str=None, limit=None):
    """
    Load events from JSONL file.

    date_str: 'YYYYMMDD' or None (for latest).
    limit: maximum number of events (None = all).

    Returns:
        events (list of dict), source_file (str or None)
    """
    fname = _choose_file_for_date(date_str)
    if fname is None:
        return [], None

    events = []
    try:
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    events.append(ev)
                except json.JSONDecodeError:
                    # skip malformed lines
                    continue
    except FileNotFoundError:
        return [], None

    if limit is not None:
        events = events[:limit]

    return events, fname


def get_summary(date_str=None):
    """
    High-level summary for dashboard/report.

    Returns:
        {
          "date": "YYYYMMDD" or None,
          "log_file": ".../events_YYYYMMDD.jsonl",
          "analysis": { ... analyze_events(...) result ... }
        }
    or None if no log file found.
    """
    events, fname = load_events(date_str=date_str)
    if fname is None:
        return None

    # infer date from filename
    base = os.path.basename(fname)           # events_YYYYMMDD.jsonl
    date_part = base.replace("events_", "").replace(".jsonl", "")

    analysis = analyze_events(events)

    return {
        "date": date_part,
        "log_file": fname,
        "analysis": analysis,
    }


def get_raw_events(date_str=None, limit=1000):
    """
    Return raw events for a given date (or latest).
    """
    events, fname = load_events(date_str=date_str, limit=limit)
    if fname is None:
        return None

    base = os.path.basename(fname)
    date_part = base.replace("events_", "").replace(".jsonl", "")

    return {
        "date": date_part,
        "log_file": fname,
        "count": len(events),
        "events": events,
    }


def get_timeline(date_str=None):
    """
    Build time-ordered event timeline for charts.

    Returns (or None if no file):
        {
          "date": "YYYYMMDD",
          "log_file": "...",
          "timeline": [
             {
               "ts": 17648xxxx.x,   # epoch seconds
               "time_iso": "...",   # ISO string
               "type": "TALKING",
               "frame": 123
             },
             ...
          ]
        }
    """
    events, fname = load_events(date_str=date_str)
    if fname is None:
        return None

    base = os.path.basename(fname)
    date_part = base.replace("events_", "").replace(".jsonl", "")

    # Sort by timestamp if present
    def _ts(ev):
        return ev.get("ts", 0)

    events_sorted = sorted(events, key=_ts)

    timeline = []
    for ev in events_sorted:
        ts = ev.get("ts", 0)
        # convert epoch to ISO if possible
        try:
            dt = datetime.utcfromtimestamp(ts)
            iso = dt.isoformat() + "Z"
        except Exception:
            iso = None

        timeline.append(
            {
                "ts": ts,
                "time_iso": iso,
                "type": ev.get("type", "UNKNOWN"),
                "frame": ev.get("frame"),
            }
        )

    return {
        "date": date_part,
        "log_file": fname,
        "timeline": timeline,
    }
