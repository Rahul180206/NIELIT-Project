# earphone_workflow.py
"""
Earphone Workflow wrapper using Roboflow inference-sdk.

Usage:
    from earphone_workflow import EarphoneWorkflow
    ew = EarphoneWorkflow()
    dets = ew.detect(frame)   # returns list of detections or [] on error
"""

import os
import tempfile
import logging
from inference_sdk import InferenceHTTPClient
import cv2
from dotenv import load_dotenv

load_dotenv()  # loads ROBOFLOW_API_KEY etc from .env if present

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EarphoneWorkflow:
    def __init__(self,
                 api_key: str = None,
                 api_url: str = "https://serverless.roboflow.com",
                 workspace_name: str = None,
                 workflow_id: str = None,
                 conf_threshold: float = 0.25,
                 timeout: int = 6):
        # allow env override
        api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        workspace_name = workspace_name or os.getenv("ROBOFLOW_WORKSPACE")
        workflow_id = workflow_id or os.getenv("ROBOFLOW_WORKFLOW")

        if not api_key or not workspace_name or not workflow_id:
            raise ValueError("EarphoneWorkflow requires api_key, workspace_name, workflow_id")

        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        self.conf_threshold = float(conf_threshold)
        self.timeout = int(timeout)

    def detect(self, frame):
        """
        Send a single frame to the workflow and return filtered detections:
        returns list of dicts: {"label","confidence","bbox":[x1,y1,x2,y2]}
        """
        # save frame to temporary path (sdk expects image path)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_name = tmp.name
        tmp.file.close()
        try:
            cv2.imwrite(tmp_name, frame)
        except Exception as e:
            logger.exception("Failed to write temp image for earphone detection: %s", e)
            return []

        try:
            result = self.client.run_workflow(
                workspace_name=self.workspace_name,
                workflow_id=self.workflow_id,
                images={"image": tmp_name},
                use_cache=True
            )
        except Exception as e:
            logger.exception("Earphone workflow request failed: %s", e)
            # cleanup file
            try:
                os.remove(tmp_name)
            except OSError:
                pass
            return []

        # remove temp file immediately
        try:
            os.remove(tmp_name)
        except OSError:
            pass

        # parse results — Roboflow returns result structure with results[0]["predictions"]
        detections = []
        try:
            logger.info("Roboflow response: %s", result)
            preds = result.get("results", [])[0].get("predictions", [])
            logger.info("Predictions found: %d", len(preds))
        except Exception as e:
            logger.error("Failed to parse Roboflow response: %s", e)
            return []

        for p in preds:
            lbl = p.get("class", "").lower()
            conf = float(p.get("confidence", 0.0))
            if conf < self.conf_threshold:
                continue
            # Roboflow's prediction provides x,y,width,height (center-based)
            cx, cy, w, h = (p.get("x", 0), p.get("y", 0), p.get("width", 0), p.get("height", 0))
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)
            detections.append({
                "label": lbl,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        return detections
