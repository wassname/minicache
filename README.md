## minicache — tiny disk cache for ML / research code.

This wraps function calls and stores returns on disk (gzip + cloudpickle). Solves
the four pain points that stdlib `functools.lru_cache + pickle` and existing
function-cache libraries (anycache, cachier) hit on ML code:

- *Loaded models can't be hashed*. So we use a arg blacklist (`exclude=["model", "tok"]`).
  Here, excluded args pass through to the function but never enter the cache key.
- *Tensors / pandas / closures can't be picked** → we use cloudpickle which extends to many more objects.
- *Pickle files grow large* → gzip on disk save 20-50%

## Quick use

Install

```sh
uv add git+https://github.com/wassname/minicache.git
```

```py
from minicache import cached, cache_call

# 1. Decorator: hashes (state, included args). Excludes drop out of key.
@cached(exclude=["model", "tok"])
def run_eval(model, tok, *, model_id, name, batch_size):
    return tinymfv_evaluate(model, tok, name=name, batch_size=batch_size)

report = run_eval(model, tok, model_id="qwen-27b", name="classic", batch_size=16)
```

See also
- anycache https://github.com/c0fec0de/anycache
- cachier https://github.com/python-cachier/cachier#working-with-unhashable-arguments
