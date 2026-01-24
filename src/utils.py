import os
import time
import random
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import joblib
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        ensure_dir(parent)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}")


@contextmanager
def timer(label: str):
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        log(f"{label} finished in {elapsed:.2f}s")


def save_model(obj: Any, path: str) -> None:
    ensure_parent_dir(path)
    joblib.dump(obj, path)
    log(f"Saved model to: {path}")


def load_model(path: str) -> Any:
    return joblib.load(path)


def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=index)
    log(f"Saved CSV to: {path}")


def save_figure(fig, path: str, dpi: int = 150) -> None:
    ensure_parent_dir(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    log(f"Saved figure to: {path}")
    fig.clf()


def make_output_dir(*parts: str) -> str:
    out_dir = os.path.join("outputs", *parts)
    ensure_dir(out_dir)
    return out_dir


def resolve_path(base: Optional[str], *parts: str) -> str:
    if base:
        return os.path.join(base, *parts)
    return os.path.join(*parts)
