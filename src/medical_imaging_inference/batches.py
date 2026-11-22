"""Small process-local queue. Only metadata is kept after each job ends."""

import uuid
from collections import OrderedDict
from threading import Lock


class BatchQueue:
    def __init__(self, pending_limit=4, history_limit=64):
        self.pending_limit = pending_limit
        self.history_limit = history_limit
        self._jobs = OrderedDict()
        self._lock = Lock()

    def reserve(self, owner, total):
        with self._lock:
            active = sum(job["status"] in {"queued", "running"} for _, job in self._jobs.values())
            if active >= self.pending_limit:
                return None
            while len(self._jobs) >= self.history_limit:
                expired = next(
                    (
                        key
                        for key, (_, job) in self._jobs.items()
                        if job["status"] not in {"queued", "running"}
                    ),
                    None,
                )
                if expired is None:
                    return None
                del self._jobs[expired]
            key = str(uuid.uuid4())
            self._jobs[key] = (owner, {"batch_id": key, "status": "queued", "total": total, "items": []})
            return key

    def update(self, key, *, status=None, item=None):
        with self._lock:
            _, job = self._jobs[key]
            if status:
                job["status"] = status
            if item is not None:
                job["items"].append(item)

    def read(self, key, owner, admin=False):
        with self._lock:
            entry = self._jobs.get(key)
            if entry is None or (entry[0] != owner and not admin):
                return None
            job = entry[1]
            return {**job, "items": list(job["items"])}
