from __future__ import annotations

import logging
import os
import time
from typing import Optional


def make_logger(logdir: str, name: str = "ld3d", log_file: Optional[str] = None) -> logging.Logger:
    """Timestamped file + console logger (avoid tqdm spam by logging only high-level events)."""
    os.makedirs(logdir, exist_ok=True)
    if log_file is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logdir, f"{name}_{ts}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"Logging to: {log_file}")
    return logger
