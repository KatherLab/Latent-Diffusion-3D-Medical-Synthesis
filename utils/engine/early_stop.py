from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 30
    min_delta: float = 0.0
    mode: str = "min"  # "min" for loss, "max" for metrics
    best: float = float("inf")
    bad_epochs: int = 0
    stopped: bool = False

    def __post_init__(self):
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        if self.mode == "max":
            self.best = float("-inf")

    def step(self, value: float) -> bool:
        improved = (value < (self.best - self.min_delta)) if self.mode == "min" else (value > (self.best + self.min_delta))
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True
        return self.stopped
