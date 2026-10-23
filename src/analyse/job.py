"""Background run of the detection pipeline, with a stop flag between tiles."""

from __future__ import annotations

import threading
import traceback
from copy import deepcopy
from typing import Any

from src.analyse.runner import PipelineStopped, run_pipeline

_LOCK = threading.Lock()
_JOB: dict[str, Any] = {
    "thread": None,
    "stop": None,
    "running": False,
    "stopping": False,
    "pending": False,
    "current": 0,
    "total": 0,
    "name": "",
    "error": "",
    "trace": "",
    "stopped": False,
    "table": None,
    "mosaic": None,
    "config": None,
    "run_id": 0,
}


def pipeline_status() -> dict[str, Any]:
    with _LOCK:
        return {
            "running": bool(_JOB["running"]),
            "stopping": bool(_JOB["stopping"]),
            "pending": bool(_JOB["pending"]),
            "current": int(_JOB["current"]),
            "total": int(_JOB["total"]),
            "name": str(_JOB["name"]),
        }


def start_pipeline(config: dict) -> None:
    if not config.get("input_dir"):
        raise ValueError("Set a tile folder under Loading.")
    snapshot = deepcopy(config)
    with _LOCK:
        thread = _JOB.get("thread")
        if thread is not None and thread.is_alive():
            raise RuntimeError("The pipeline is already running.")
        stop = threading.Event()
        run_id = int(_JOB["run_id"]) + 1
        _JOB["stop"] = stop
        _JOB["running"] = True
        _JOB["stopping"] = False
        _JOB["pending"] = False
        _JOB["current"] = 0
        _JOB["total"] = 0
        _JOB["name"] = "Starting…"
        _JOB["error"] = ""
        _JOB["trace"] = ""
        _JOB["stopped"] = False
        _JOB["table"] = None
        _JOB["mosaic"] = None
        _JOB["config"] = snapshot
        _JOB["run_id"] = run_id
        worker = threading.Thread(
            target=_pipeline_thread,
            args=(snapshot, stop, run_id),
            name="pipeline",
            daemon=True,
        )
        _JOB["thread"] = worker
    worker.start()


def stop_pipeline() -> None:
    with _LOCK:
        stop = _JOB.get("stop")
        if stop is not None:
            stop.set()
        if _JOB["running"]:
            _JOB["stopping"] = True


def claim_pipeline_result() -> dict[str, Any] | None:
    with _LOCK:
        if _JOB["running"] or not _JOB["pending"]:
            return None
        _JOB["pending"] = False
        result = {
            "stopped": bool(_JOB["stopped"]),
            "error": str(_JOB["error"]),
            "trace": str(_JOB["trace"]),
            "table": _JOB["table"],
            "mosaic": _JOB["mosaic"],
            "config": _JOB["config"],
        }
        _JOB["table"] = None
        _JOB["mosaic"] = None
        return result


def _pipeline_thread(config: dict, stop: threading.Event, run_id: int) -> None:
    def _on_progress(current: int, total: int, name: str) -> None:
        with _LOCK:
            if _JOB["run_id"] != run_id:
                return
            _JOB["current"] = int(current)
            _JOB["total"] = int(total)
            _JOB["name"] = str(name)

    try:
        table, mosaic = run_pipeline(
            config,
            progress_cb=_on_progress,
            should_stop=stop.is_set,
        )
    except PipelineStopped:
        _finish(run_id, stopped=True)
        return
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        _finish(run_id, error=str(exc))
        return
    except Exception as exc:
        _finish(run_id, error=str(exc), trace=traceback.format_exc())
        return
    _finish(run_id, table=table, mosaic=mosaic, config=config)


def _finish(
    run_id: int,
    *,
    stopped: bool = False,
    error: str = "",
    trace: str = "",
    table: Any = None,
    mosaic: Any = None,
    config: dict | None = None,
) -> None:
    with _LOCK:
        if _JOB["run_id"] != run_id:
            return
        _JOB["running"] = False
        _JOB["stopping"] = False
        _JOB["pending"] = True
        _JOB["stopped"] = stopped
        _JOB["error"] = error
        _JOB["trace"] = trace
        _JOB["table"] = table
        _JOB["mosaic"] = mosaic
        if config is not None:
            _JOB["config"] = config
