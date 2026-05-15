"""minicache — tiny disk cache for ML / research code.

Wraps function calls and stores returns on disk (gzip + cloudpickle). Solves
the four pain points that stdlib `functools.lru_cache + pickle` and existing
function-cache libraries (anycache, cachier) hit on ML code:

- **Loaded models can't be hashed** → arg blacklist (`exclude=["model", "tok"]`).
  Excluded args pass through to the function but never enter the cache key.
- **Tensors / pandas / closures break stdlib pickle** → cloudpickle backend.
- **Pickle files grow large** → gzip on disk (~3× smaller, free).
- **"Function source changed → invalidate" causes false invalidations on
  reformat** → caller bumps an explicit `state` string when behavior actually
  changes. No AST hashing magic.

## Quick use

    from minicache import cached, cache_call

    # 1. Decorator: hashes (state, included args). Excludes drop out of key.
    @cached("eval", cachedir="out/cache",
            state_fn=lambda *, model_id, **_: f"{model_id}|nf4|r00+r02",
            exclude=["model", "tok"])
    def run_eval(model, tok, *, model_id, name, batch_size):
        return tinymfv_evaluate(model, tok, name=name, batch_size=batch_size)

    report = run_eval(model, tok, model_id="qwen-27b", name="classic", batch_size=16)

    # 2. Explicit key: no introspection, you compose the key
    key = "qwen-27b|nf4|r00+r02|eval|classic|bs=16"
    report = cache_call("eval", key, lambda: tinymfv_evaluate(model, tok, ...),
                        cachedir="out/cache")

See also
- anycache https://github.com/c0fec0de/anycache
- cachier https://github.com/python-cachier/cachier#working-with-unhashable-arguments
"""
from __future__ import annotations

import gzip
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import cloudpickle


__version__ = "0.1.0"
__all__ = ["cache_call", "cached"]

_EXT = ".pkl.gz"


def _hash(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_call(kind: str, key: str, fn: Callable[[], Any],
               cachedir: Path | str) -> Any:
    """Run-or-load. Cache file = `<cachedir>/<kind>/<key>.pkl.gz`.

    Hit: gunzip + cloudpickle.load → return.
    Miss: run fn(), gunzip-pickle the return, return.
    No silent fallbacks: corrupt cache or unpicklable return value raises.
    """
    p = Path(cachedir) / kind / f"{key}{_EXT}"
    if p.exists():
        with gzip.open(p, "rb") as f:
            return cloudpickle.load(f)
    result = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wb") as f:
        cloudpickle.dump(result, f)
    return result


def cached(
    kind: str,
    *,
    cachedir: Path | str,
    state_fn: Callable[..., str] | None = None,
    exclude: Iterable[str] = (),
):
    """Decorator. Cache key = sha256(kind | state_fn(**args) | included_args)
    where included = signature(fn) \\ exclude.

    `state_fn` lets you inject context that isn't a function arg (e.g. a model
    fingerprint walked from disk). It receives ALL bound args by name; pull
    out what you need with **kwargs unpacking.

    Args in `exclude` pass through to fn unchanged but never enter the key —
    use this for unhashable / large / instance-specific things (loaded models,
    open files, GPU tensors).
    """
    excluded = set(exclude)

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        keep = [n for n in sig.parameters if n not in excluded]

        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            included = {n: bound.arguments[n] for n in keep
                        if n in bound.arguments}
            state = state_fn(**bound.arguments) if state_fn else ""
            payload = json.dumps({"k": kind, "s": state, "a": included},
                                 sort_keys=True, default=str)
            key = _hash(payload)
            return cache_call(kind, key, lambda: fn(*args, **kwargs), cachedir)

        wrapper.__wrapped__ = fn
        return wrapper
    return decorator


if __name__ == "__main__":
    # Smoke: hits cache on second call.
    import tempfile
    import time
    with tempfile.TemporaryDirectory() as td:
        @cached("demo", cachedir=td, exclude=["expensive_obj"])
        def f(x, expensive_obj=None, y=10):
            time.sleep(0.5)
            return x + y

        t0 = time.time(); assert f(1, expensive_obj=object()) == 11
        t1 = time.time(); assert f(1, expensive_obj=object()) == 11  # cache HIT
        t2 = time.time()
        print(f"miss: {t1-t0:.3f}s, hit: {t2-t1:.4f}s (different obj instance, same key)")

        report = cache_call("eval", "test-key", lambda: {"a": 1, "b": [2, 3]}, cachedir=td)
        assert report == {"a": 1, "b": [2, 3]}
        print("explicit-key OK")
