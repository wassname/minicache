"""minicache — tiny disk cache for ML / research code.

Wraps function calls and stores returns on disk (gzip + cloudpickle). Solves
four pain points that `functools.lru_cache + pickle` and existing function-cache
libs (anycache, cachier) hit on ML code:

- **Loaded models can't be hashed** → arg blacklist (`exclude=["model", "tok"]`).
  Excluded args pass through to the function but never enter the cache key.
- **Tensors / pandas / closures break stdlib pickle** → cloudpickle backend.
- **Pickle files grow large** → gzip on disk (~3× smaller, free).
- **No source-AST hashing** → false invalidation on reformat is the worst kind
  of bug. Caller passes a `state` kwarg (or anything else) when behavior
  changes. No magic.

## Usage

    from minicache import cached, cache_call

    # Decorator. Default cachedir = ./cache, default kind = fn.__name__.
    @cached(exclude=["model", "tok"])
    def run_eval(model, tok, *, model_id, name, batch_size):
        return expensive_eval(model, tok, name=name, batch_size=batch_size)

    report = run_eval(model, tok, model_id="qwen-27b", name="classic", batch_size=16)
    # second call with same args (any model/tok instance) → cache HIT

    # Explicit-key form. Caller composes the key (no introspection).
    # Useful when args alone don't determine the cache identity (e.g. you
    # also want to pin to disk state walked at call time).
    report = cache_call("eval", "qwen-27b|nf4|r00+r02|classic|bs=16",
                        lambda: expensive_eval(model, tok, name="classic"))

See also:
- anycache https://github.com/c0fec0de/anycache
- cachier https://github.com/python-cachier/cachier
"""
from __future__ import annotations

import gzip
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import cloudpickle


__version__ = "0.2.0"
__all__ = ["cache_call", "cached", "DEFAULT_CACHEDIR"]

DEFAULT_CACHEDIR = Path("cache")
_EXT = ".pkl.gz"


def _hash(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_call(kind: str, key: str, fn: Callable[[], Any],
               cachedir: Path | str = DEFAULT_CACHEDIR) -> Any:
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
    *,
    exclude: Iterable[str] = (),
    cachedir: Path | str = DEFAULT_CACHEDIR,
    kind: str | None = None,
):
    """Decorator. Cache key = sha256(json(included args)).

    `exclude` drops args from the key — use for unhashable / large /
    instance-specific values (loaded models, GPU tensors, open files). They
    still pass through to the function unchanged.

    `kind` is the cache subdir (default = fn.__name__).
    """
    excluded = set(exclude)

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        keep = [n for n in sig.parameters if n not in excluded]
        sub = kind or fn.__name__

        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            included = {n: bound.arguments[n] for n in keep
                        if n in bound.arguments}
            payload = json.dumps(included, sort_keys=True, default=str)
            key = _hash(payload)
            return cache_call(sub, key, lambda: fn(*args, **kwargs), cachedir)

        wrapper.__wrapped__ = fn
        return wrapper
    return decorator


if __name__ == "__main__":
    import tempfile
    import time
    with tempfile.TemporaryDirectory() as td:
        @cached(exclude=["expensive_obj"], cachedir=td)
        def f(x, expensive_obj=None, y=10):
            time.sleep(0.5)
            return x + y

        t0 = time.time(); assert f(1, expensive_obj=object()) == 11
        t1 = time.time(); assert f(1, expensive_obj=object()) == 11
        t2 = time.time()
        print(f"miss: {t1-t0:.3f}s, hit: {t2-t1:.4f}s (different obj instance, same key)")

        report = cache_call("eval", "test-key", lambda: {"a": 1, "b": [2, 3]}, cachedir=td)
        assert report == {"a": 1, "b": [2, 3]}
        print("explicit-key OK")
