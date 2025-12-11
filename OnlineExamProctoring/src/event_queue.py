# event_queue.py
import threading
import time
import queue
import requests
import json

class EventSender:
    def __init__(self, base_url, endpoint, batch_size=10,
                 send_interval=1.0, max_queue_size=2000):

        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.url = self.base_url + endpoint

        self.batch_size = batch_size
        self.send_interval = send_interval
        self.queue = queue.Queue(maxsize=max_queue_size)

        self.running = False
        self.thread = None

    # ---------------------------------------------------------
    def enqueue(self, event):
        """Try to add event to queue. Drop if full."""
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            print("[EventSender] WARNING: queue full, dropping event")

    # ---------------------------------------------------------
    def start(self):
        """Start background sending thread."""
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print(f"[EventSender] Started worker thread, sending to {self.url}")

    # ---------------------------------------------------------
    def stop(self):
        """Stop sender cleanly."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        print("[EventSender] Stopped sender")

    # ---------------------------------------------------------
    def _worker(self):
        """Background task sending batched events."""
        while self.running:
            batch = []
            try:
                # Wait for first event or timeout
                ev = self.queue.get(timeout=self.send_interval)
                batch.append(ev)
            except queue.Empty:
                continue

            # Collect remaining events up to batch_size
            while len(batch) < self.batch_size:
                try:
                    ev = self.queue.get_nowait()
                    batch.append(ev)
                except queue.Empty:
                    break

            # Send batch to backend
            try:
                response = requests.post(
                    self.url,
                    json=batch,
                    timeout=2
                )
                print(f"[EventSender] Sent batch of {len(batch)} events → status {response.status_code}")
            except Exception as e:
                print("[EventSender] ERROR sending batch:", e)

            time.sleep(self.send_interval)
