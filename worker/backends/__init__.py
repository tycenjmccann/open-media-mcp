"""Model backends. Each module exposes BACKEND, an object with:

  name, license, notice        identity + the attribution the license asks for (shown in job results)
  fps, models                  output frame rate; [(hf_repo, file, comfy_models_subdir)] fetched at boot
  frames(seconds) -> int       snap a length to the model's frame cadence
  build_graph(spec, prep, names, prefix) -> dict   ComfyUI API-format graph

`spec` is the normalized job (see lambda/tools/backends.json for the per-backend limits the tools Lambda enforces).
`prep` holds the locally prepared media; `names` maps each local path to its uploaded ComfyUI input name.
Add a backend = add a module here + an entry in lambda/tools/backends.json + KNOWN_BACKENDS in the stack.
"""
import importlib

MODULES = {"wan2.2": "wan22", "h3": "h3"}


def load(names):
    out = {}
    for n in names:
        if n not in MODULES:
            raise ValueError(f"unknown backend {n}")
        out[n] = importlib.import_module(f"{__name__}.{MODULES[n]}").BACKEND
    return out
