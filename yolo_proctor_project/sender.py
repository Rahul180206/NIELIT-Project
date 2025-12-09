"""
sender.py
Responsible for sending event JSON to backend and uploading evidence images.
Implements retries with exponential backoff (3 attempts).
"""

import os
import json
import time
import logging
from typing import Dict, Optional
import requests

logger = logging.getLogger(__name__)


def send_event(event_json: Dict, backend_url: str, timeout: float = 5.0, max_retries: int = 3) -> Optional[requests.Response]:
    """
    Send JSON event to backend (application/json).
    Retries with exponential backoff on non-200 / exceptions.
    """
    url = backend_url
    headers = {"Content-Type": "application/json"}
    data = json.dumps(event_json)
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, data=data, headers=headers, timeout=timeout)
            if resp.status_code >= 200 and resp.status_code < 300:
                logger.info("Event posted successfully: %s", resp.status_code)
                return resp
            else:
                logger.warning("Backend returned status %s: %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.warning("Send attempt %d failed: %s", attempt, exc)
        time.sleep(backoff)
        backoff *= 2.0
    logger.error("Failed to POST event after %d attempts", max_retries)
    return None


def upload_evidence(event_json: Dict, image_path: str, backend_url: str, timeout: float = 10.0, max_retries: int = 3) -> Optional[requests.Response]:
    """
    Upload evidence (multipart/form-data) to backend_url?upload=true
    Fields:
      - json: JSON string
      - file: image binary (open)
    Returns response or None.
    """
    url = backend_url
    if "?" in url:
        url_upload = url + "&upload=true"
    else:
        url_upload = url + "?upload=true"
    files = {
        "json": (None, json.dumps(event_json), "application/json"),
        "file": (os.path.basename(image_path), open(image_path, "rb"), "image/jpeg")
    }
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url_upload, files=files, timeout=timeout)
            if r.status_code >= 200 and r.status_code < 300:
                logger.info("Evidence uploaded: %s", r.status_code)
                return r
            else:
                logger.warning("Upload returned status %s: %s", r.status_code, r.text)
        except Exception as exc:
            logger.warning("Upload attempt %d failed: %s", attempt, exc)
        time.sleep(backoff)
        backoff *= 2.0
    logger.error("Failed to upload evidence after %d attempts", max_retries)
    return None
