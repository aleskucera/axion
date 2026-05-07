"""Tiny FastAPI server that exposes axion HDF5 logs as JSON for the
Observable Plot frontend.

Install:
    uv pip install fastapi uvicorn

Run from repo root:
    uvicorn tools.dashboard.server:app --reload --port 8000

Then open http://localhost:8000
"""
from pathlib import Path

import h5py
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = ROOT / "data" / "logs"
HERE = Path(__file__).resolve().parent

app = FastAPI(title="axion convergence")


def _to_jsonable(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _safe_path(filename: str) -> Path:
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "invalid filename")
    p = LOGS_DIR / filename
    if not p.is_file():
        raise HTTPException(404, f"file not found: {filename}")
    return p


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/runs")
def list_runs():
    if not LOGS_DIR.is_dir():
        return []
    runs = []
    for p in LOGS_DIR.glob("*.h5"):
        runs.append({"name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)})
    runs.sort(key=lambda r: r["name"], reverse=True)
    return runs


@app.get("/api/run/{filename}/meta")
def meta(filename: str):
    p = _safe_path(filename)
    with h5py.File(p, "r") as f:
        out = {k: _to_jsonable(v) for k, v in f.attrs.items()}
        out["has_pcr_history"] = "pcr_history_res_norm_sq_history" in f
        out["has_linesearch"] = "ls_history_minimal_index" in f
        if "iter_count" in f:
            out["num_steps"] = int(f["iter_count"].shape[0])
        if "candidates_res_norm_sq" in f:
            out["max_newton_iters"] = int(f["candidates_res_norm_sq"].shape[1])
            out["num_worlds"] = int(f["candidates_res_norm_sq"].shape[2])
        if "pcr_history_res_norm_sq_history" in f:
            out["max_linear_iters_p1"] = int(
                f["pcr_history_res_norm_sq_history"].shape[2]
            )
        if "body_pose" in f:
            out["body_count"] = int(f["body_pose"].shape[2])
        return out


@app.get("/api/run/{filename}/convergence")
def convergence(filename: str, world: int = 0):
    """Per-step summary: NR iters used and final NR ‖r‖²."""
    p = _safe_path(filename)
    with h5py.File(p, "r") as f:
        if "iter_count" not in f or "candidates_res_norm_sq" not in f:
            raise HTTPException(400, "file missing iter_count/candidates_res_norm_sq")
        nr_iters = f["iter_count"][:, 0].astype(int)
        cand = f["candidates_res_norm_sq"]
        S, K, W = cand.shape
        if world < 0 or world >= W:
            raise HTTPException(400, f"world out of range (0..{W-1})")
        nr_res = cand[:, :, world]
        last_idx = np.maximum(nr_iters - 1, 0)
        final = nr_res[np.arange(S), last_idx]
        max_K = K
        return {
            "step": list(range(S)),
            "nr_iters": nr_iters.tolist(),
            "final_residual_sq": [float(x) for x in final],
            "max_newton_iters": max_K,
        }


@app.get("/api/run/{filename}/heatmap")
def heatmap(filename: str, world: int = 0):
    """Per-(step, NR-iter) NR ‖r‖² for a heatmap.

    Returns flattened rows {step, iter, log_res, ran}. Iters that did not
    actually execute (k >= iter_count[step]) are marked ran=False so the
    frontend can blank them.
    """
    p = _safe_path(filename)
    with h5py.File(p, "r") as f:
        if "iter_count" not in f or "candidates_res_norm_sq" not in f:
            raise HTTPException(400, "file missing iter_count/candidates_res_norm_sq")
        nr_iters = f["iter_count"][:, 0].astype(int)
        cand = f["candidates_res_norm_sq"][:, :, world]
        S, K = cand.shape
        ran = (np.arange(K)[None, :] < nr_iters[:, None]).astype(bool)
        log_res = np.log10(np.maximum(cand, 1e-30))
        rows = []
        for s in range(S):
            for k in range(K):
                if ran[s, k]:
                    rows.append({"step": int(s), "iter": int(k),
                                 "log_res": float(log_res[s, k])})
        return {
            "rows": rows,
            "max_step": int(S - 1),
            "max_iter": int(K - 1),
        }


def _residual_offsets(f: h5py.File):
    """Return slice ranges for each block of the residual vector.

    Standard layout in `_res` (and `_candidates_res`):
        [0, N_u)           dynamics  (6 floats per body)
        [N_u, N_u+N_j)     joint
        [+N_ctrl)          control
        [+N_n)             contact normal
        [+N_f)             contact friction
    """
    if "dims" not in f:
        raise HTTPException(400, "file missing 'dims' group with offset attrs")
    d = f["dims"].attrs
    N_u, N_j, N_ctrl, N_n, N_f = (
        int(d["N_u"]), int(d["N_j"]), int(d["N_ctrl"]),
        int(d["N_n"]), int(d["N_f"]),
    )
    o0 = 0
    o1 = N_u
    o2 = o1 + N_j
    o3 = o2 + N_ctrl
    o4 = o3 + N_n
    o5 = o4 + N_f
    return [
        ("dynamics", o0, o1),
        ("joint",    o1, o2),
        ("control",  o2, o3),
        ("normal",   o3, o4),
        ("friction", o4, o5),
    ]


def _block_sq_norms(arr_2d: np.ndarray, blocks):
    """arr_2d: (..., N_total). Returns dict name -> (...,) of ‖block‖²."""
    out = {}
    for name, lo, hi in blocks:
        out[name] = np.sum(arr_2d[..., lo:hi] ** 2, axis=-1)
    return out


@app.get("/api/run/{filename}/decomposition")
def decomposition(filename: str, world: int = 0):
    """Per-step residual decomposition at the FINAL NR iter of each step.

    Returns step + per-block ‖r‖² so the frontend can stack them.
    """
    p = _safe_path(filename)
    with h5py.File(p, "r") as f:
        if "_candidates_res" not in f or "iter_count" not in f:
            raise HTTPException(400, "file missing _candidates_res / iter_count")
        cand_res = f["_candidates_res"]                  # (S, K, W, N)
        nr_iters = f["iter_count"][:, 0].astype(int)
        S, K, W, N = cand_res.shape
        last_idx = np.maximum(nr_iters - 1, 0)
        # Read just the per-step final-iter slice. h5py allows indexing
        # along axis 0 with arrays via fancy access, but it's simpler to
        # do a loop here since S = 200 typical.
        final_per_step = np.empty((S, N), dtype=np.float64)
        for s in range(S):
            final_per_step[s] = cand_res[s, last_idx[s], world]
        blocks = _residual_offsets(f)
        norms = _block_sq_norms(final_per_step, blocks)
        out = {"step": list(range(S))}
        for name, _, _ in blocks:
            out[name] = [float(x) for x in norms[name]]
        return out


@app.get("/api/run/{filename}/step/{step}")
def step_detail(filename: str, step: int, world: int = 0):
    """Per-NR-iter detail for a chosen step."""
    p = _safe_path(filename)
    with h5py.File(p, "r") as f:
        if "iter_count" not in f:
            raise HTTPException(400, "file missing iter_count")
        S = int(f["iter_count"].shape[0])
        if step < 0 or step >= S:
            raise HTTPException(400, f"step out of range (0..{S-1})")
        max_K = int(f["candidates_res_norm_sq"].shape[1])
        nr_iters_used = int(f["iter_count"][step, 0])

        out = {
            "step": step,
            "world": world,
            "nr_iters_used": nr_iters_used,
            "max_newton_iters": max_K,
            "nr_residual_sq": f["candidates_res_norm_sq"][step, :, world]
            .astype(float)
            .tolist(),
        }

        if "pcr_history_iter_count" in f:
            out["pcr_iters_per_nr"] = (
                f["pcr_history_iter_count"][step, :, 0].astype(int).tolist()
            )

        if "pcr_history_res_norm_sq_history" in f:
            pcr = f["pcr_history_res_norm_sq_history"][step, :, :, world]
            out["pcr_residual_history"] = pcr.astype(float).tolist()

        if "ls_history_minimal_index" in f and "ls_history_step_size" in f:
            ls_idx = f["ls_history_minimal_index"][step, :, world].astype(int)
            ls_grid = f["ls_history_step_size"][step]  # (K, A)
            chosen = []
            for k in range(max_K):
                row = ls_grid[k] if ls_grid.ndim == 2 else ls_grid
                chosen.append(float(row[ls_idx[k]]))
            out["chosen_alpha"] = chosen

        if "_candidates_res" in f and "dims" in f:
            blocks = _residual_offsets(f)
            cand_res = f["_candidates_res"][step, :, world, :]  # (K, N)
            norms = _block_sq_norms(cand_res, blocks)
            out["decomposition"] = {
                name: [float(x) for x in norms[name]]
                for name, _, _ in blocks
            }

        return out
