#!/usr/bin/env python3
"""VarStar Lab - web app: astronomical images in, light curve, period, distance out.

Built for the project proposal "Photographing and Modeling Variable Stars"
(Prasarttongosoth, Hawkins & Collar), whose target V0756 CrA is a high-amplitude
delta Scuti - a class that obeys a period-luminosity relation, which is what makes
a distance obtainable from a period.

Run:  python3 app.py           (serves on http://0.0.0.0:12113)
"""
from __future__ import annotations

import io
import json
import os
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, Response, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import distance as dist_mod
from pipeline import ingest, period as per_mod, photometry as phot_mod, plotting as pl
from pipeline import synth, timing

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
STATIC = os.path.join(BASE, "static")
os.makedirs(DATA, exist_ok=True)

PORT = int(os.environ.get("VARSTAR_PORT", "12113"))
MAX_FRAMES = int(os.environ.get("VARSTAR_MAX_FRAMES", "1200"))
MAX_UPLOAD_MB = int(os.environ.get("VARSTAR_MAX_UPLOAD_MB", "4096"))
SESSION_TTL = 6 * 3600
MAX_SESSIONS = 24

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Session state is in memory, so any directory left on disk from a previous
    # run is unreachable. Clear it before serving.
    removed, freed = purge_orphan_dirs()
    if removed:
        print(f"cleaned {removed} orphaned session dir(s), freed "
              f"{freed / (1 << 30):.2f} GiB")
    yield


app = FastAPI(title="VarStar Lab", docs_url="/api/docs", redoc_url=None,
              lifespan=lifespan)
POOL = ThreadPoolExecutor(max_workers=4)
_LOCK = threading.Lock()
SESSIONS: dict[str, "Session"] = {}


# ===========================================================================
# session state
# ===========================================================================

class Session:
    def __init__(self, sid: str):
        self.id = sid
        self.dir = os.path.join(DATA, sid)
        self.frames_dir = os.path.join(self.dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.created = time.time()
        self.touched = time.time()

        self.frames: list[ingest.Frame] = []
        self.synth_manifest: Optional[dict] = None
        self.sources: list[dict] = []
        self.ref_index: int = 0
        self.phot_cfg = phot_mod.PhotConfig()
        self.phot: Optional[phot_mod.PhotResult] = None
        self.diff: Optional[dict] = None
        self.target: Optional[int] = None
        self.comps: list[int] = []
        self.comp_mags: dict = {}
        self.period_result: Optional[dict] = None
        self.distance_result: Optional[dict] = None
        self.gaia: Optional[dict] = None
        self.target_info = dict(timing.TARGET_DEFAULT)
        self.time_label = "JD (UTC)"
        self.times: Optional[np.ndarray] = None
        self.job = {"running": False, "stage": "", "current": 0, "total": 0,
                    "message": "", "error": None, "done": False}
        self.plots: dict = {}
        self.log: list[str] = []

    def touch(self):
        self.touched = time.time()

    def note(self, msg: str):
        self.log.append(f"{time.strftime('%H:%M:%S')}  {msg}")
        del self.log[:-200]

    def set_job(self, **kw):
        self.job.update(kw)

    def progress(self, i, total, msg=""):
        self.job.update({"current": int(i), "total": int(total), "message": str(msg)})

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    @property
    def usable_frames(self):
        return [f for f in self.frames if not f.excluded]


def get_session(sid: str) -> Session:
    with _LOCK:
        s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "Session not found or expired. Reload the page to "
                                 "start a new one.")
    s.touch()
    return s


def reap():
    now = time.time()
    with _LOCK:
        dead = [k for k, s in SESSIONS.items()
                if now - s.touched > SESSION_TTL and not s.job["running"]]
        if len(SESSIONS) - len(dead) > MAX_SESSIONS:
            alive = sorted((s for k, s in SESSIONS.items() if k not in dead),
                           key=lambda s: s.touched)
            dead += [s.id for s in alive[:len(SESSIONS) - len(dead) - MAX_SESSIONS]]
        for k in set(dead):
            s = SESSIONS.pop(k, None)
            if s:
                s.cleanup()


def purge_orphan_dirs():
    """Delete on-disk session directories with no live session.

    Session state lives in memory, so every directory present at startup is
    unreachable by definition. Without this, a few hundred frames per run times
    a few restarts silently fills the disk.
    """
    if not os.path.isdir(DATA):
        return 0, 0
    freed = removed = 0
    with _LOCK:
        live = set(SESSIONS)
    for name in os.listdir(DATA):
        if name in live:
            continue
        path = os.path.join(DATA, name)
        if not os.path.isdir(path):
            continue
        try:
            for root, _, files in os.walk(path):
                for fn in files:
                    try:
                        freed += os.path.getsize(os.path.join(root, fn))
                    except OSError:
                        pass
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        except OSError:
            pass
    return removed, freed


def busy_guard(s: Session):
    if s.job["running"]:
        raise HTTPException(409, f"Still working: {s.job['stage']}. Wait for it to "
                                 "finish or reload the page.")


def run_bg(s: Session, stage: str, fn):
    """Run `fn` on the pool, tracking progress and surfacing errors to the UI."""
    busy_guard(s)
    s.set_job(running=True, stage=stage, current=0, total=0, message="starting",
              error=None, done=False)

    def wrapper():
        try:
            fn()
            s.set_job(running=False, done=True, message="complete")
        except Exception as exc:
            tb = traceback.format_exc(limit=6)
            s.note(f"ERROR in {stage}: {exc}")
            s.set_job(running=False, done=True, error=str(exc), message="failed")
            s.job["traceback"] = tb

    POOL.submit(wrapper)
    return {"started": True, "stage": stage}


# ===========================================================================
# request models
# ===========================================================================

class DemoReq(BaseModel):
    preset: str = "single_night"
    n_frames: Optional[int] = None
    n_nights: Optional[int] = None
    cadence: Optional[float] = None
    exptime: Optional[float] = None
    period: Optional[float] = None
    amplitude: Optional[float] = None
    mean_mag: Optional[float] = None
    seed: Optional[int] = None
    noise: Optional[bool] = None
    second_mode: Optional[bool] = None


class TimesReq(BaseModel):
    start_utc: str
    cadence_s: float
    exptime_s: Optional[float] = None


class DetectReq(BaseModel):
    frame: int = 0
    fwhm: float = 4.0
    thresh_sigma: float = 5.0
    channel: str = "G"
    max_sources: int = 60


class PhotReq(BaseModel):
    target: int
    comps: list[int]
    comp_mags: dict = Field(default_factory=dict)
    channel: str = "G"
    fwhm: float = 4.0
    ap_factor: float = 1.5
    ann_in_factor: float = 3.0
    ann_out_factor: float = 5.0
    gain: float = 1.0
    read_noise: float = 0.0
    track: bool = True
    global_align: bool = True
    track_box: int = 11
    mean_mag_manual: Optional[float] = None


class PeriodReq(BaseModel):
    p_min: float = 0.02
    p_max: float = 0.30
    oversample: int = 25
    nharm: int = 3
    detrend_order: int = -1
    n_modes: int = 3
    bootstrap: int = 150
    pdm_bins: int = 10
    pdm_periods: int = 3000
    time_system: str = "bjd_tdb"
    ra: Optional[str] = None
    dec: Optional[str] = None
    site_lat: Optional[float] = None
    site_lon: Optional[float] = None
    site_elev: Optional[float] = None
    force_period: Optional[float] = None
    catalog_period: Optional[float] = None


class DistReq(BaseModel):
    relation: str = "ziaali2019"
    period_source: str = "measured"     # measured | catalog | manual
    manual_period: Optional[float] = None
    mean_mag: Optional[float] = None
    sigma_mean_mag: Optional[float] = None
    ebv: Optional[float] = None
    a_v: Optional[float] = None
    sigma_ext: Optional[float] = None
    custom_slope: Optional[float] = None
    custom_intercept: Optional[float] = None
    custom_slope_err: Optional[float] = None
    custom_intercept_err: Optional[float] = None
    custom_scatter: Optional[float] = None
    teff: Optional[float] = None
    use_fundamental: bool = True


class LookupReq(BaseModel):
    ra: Optional[str] = None
    dec: Optional[str] = None


# ===========================================================================
# helpers
# ===========================================================================

def jsonable(obj):
    """Make numpy types JSON-safe, turning non-finite floats into None."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def ok(payload):
    return JSONResponse(jsonable(payload))


def frames_payload(s: Session):
    n_time = sum(1 for f in s.frames if f.jd is not None)
    jds = [f.jd for f in s.frames if f.jd is not None]
    span = (max(jds) - min(jds)) if len(jds) > 1 else 0.0
    kinds = sorted({f.kind for f in s.frames})
    sessions, _ = per_mod.session_stats(np.array(jds)) if len(jds) > 1 else ([], None)
    return {
        "n_frames": len(s.frames),
        "n_usable": len(s.usable_frames),
        "n_with_time": n_time,
        "needs_manual_times": n_time < len(s.frames) and len(s.frames) > 0,
        "span_days": span, "span_hours": span * 24.0,
        "kinds": kinds,
        "n_sessions": max(1, len(sessions)) if jds else 0,
        "sessions": sessions,
        "is_synthetic": s.synth_manifest is not None,
        "synthetic_truth": ({k: v for k, v in s.synth_manifest["truth"].items()
                             if k not in ("rows", "stars", "stars_frame1")}
                            if s.synth_manifest else None),
        "frames": [f.to_dict() for f in s.frames[:MAX_FRAMES]],
    }


# ===========================================================================
# routes - session & ingest
# ===========================================================================

@app.post("/api/session")
def create_session():
    reap()
    purge_orphan_dirs()
    sid = uuid.uuid4().hex[:16]
    s = Session(sid)
    with _LOCK:
        SESSIONS[sid] = s
    s.note("session created")
    return ok({"session_id": sid, "target_default": s.target_info,
               "presets": {k: {"label": v["label"], "description": v["description"]}
                           for k, v in synth.PRESETS.items()},
               "relations": {k: {"label": v["label"], "note": v["note"],
                                 "slope": v["slope"], "intercept": v["intercept"],
                                 "scatter": v["scatter"], "valid": v["valid"]}
                             for k, v in dist_mod.PL_RELATIONS.items()}})


@app.get("/api/session/{sid}")
def session_state(sid: str):
    s = get_session(sid)
    return ok({
        "session_id": sid,
        "frames": frames_payload(s),
        "job": s.job,
        "has_photometry": s.phot is not None,
        "has_period": s.period_result is not None,
        "has_distance": s.distance_result is not None,
        "target": s.target, "comps": s.comps,
        "n_sources": len(s.sources),
        "log": s.log[-40:],
    })


@app.get("/api/session/{sid}/job")
def job_state(sid: str):
    s = get_session(sid)
    return ok(s.job)


@app.post("/api/session/{sid}/upload")
async def upload(sid: str, files: list[UploadFile] = File(...)):
    s = get_session(sid)
    busy_guard(s)
    if len(s.frames) + len(files) > MAX_FRAMES:
        raise HTTPException(413, f"Too many frames (limit {MAX_FRAMES}).")

    total_bytes = 0
    saved = []
    for up in files:
        name = os.path.basename(up.filename or "frame")
        if ingest.classify(name) == "unknown":
            continue
        dest = os.path.join(s.frames_dir, f"{len(saved) + len(s.frames):05d}_{name}")
        size = 0
        with open(dest, "wb") as fh:
            while chunk := await up.read(1 << 20):
                size += len(chunk)
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_MB * (1 << 20):
                    fh.close()
                    os.unlink(dest)
                    raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_MB} MB.")
                fh.write(chunk)
        saved.append(dest)
    if not saved:
        raise HTTPException(400,
                            "No readable image files found. Accepted: FITS (.fits/.fit/.fts), "
                            "TIFF, PNG, JPEG, and camera RAW (.cr2/.cr3/.nef/.arw/.dng). "
                            "FITS is strongly preferred - it keeps the linear pixel values "
                            "that photometry needs.")

    s.synth_manifest = None

    def work():
        s.set_job(stage="reading frame metadata", total=len(saved))
        start = len(s.frames)
        for i, path in enumerate(saved):
            s.progress(i, len(saved), os.path.basename(path))
            s.frames.append(ingest.scan_frame(path, start + i))
        _post_ingest(s)

    return ok(run_bg(s, "reading frame metadata", work))


def _post_ingest(s: Session):
    """Order frames in time, seed sensible defaults from the headers."""
    have = [f for f in s.frames if f.jd is not None]
    if len(have) == len(s.frames) and s.frames:
        s.frames.sort(key=lambda f: f.jd)
    elif s.frames and not have:
        # No timestamps at all: fall back to filename sequence order.
        ingest.times_from_filenames(s.frames)
    for i, f in enumerate(s.frames):
        f.index = i

    gains = [f.gain_hdr for f in s.frames if f.gain_hdr]
    if gains:
        s.phot_cfg.gain = float(np.median(gains))
    sats = [f.saturation for f in s.frames if f.saturation]
    if sats:
        s.phot_cfg.saturation = float(np.median(sats))
    s.note(f"{len(s.frames)} frames ingested; "
           f"{sum(1 for f in s.frames if f.jd is not None)} with timestamps")


@app.post("/api/session/{sid}/demo")
def make_demo(sid: str, req: DemoReq):
    s = get_session(sid)
    busy_guard(s)
    overrides = {k: v for k, v in req.model_dump().items()
                 if k != "preset" and v is not None}
    cfg = synth.config_from_preset(req.preset, **overrides)
    if cfg.n_frames > MAX_FRAMES:
        raise HTTPException(400, f"Frame count capped at {MAX_FRAMES}.")

    def work():
        shutil.rmtree(s.frames_dir, ignore_errors=True)
        os.makedirs(s.frames_dir, exist_ok=True)
        s.frames = []
        s.sources = []
        s.phot = s.diff = s.period_result = s.distance_result = None
        s.set_job(stage="simulating exposures", total=cfg.n_frames)
        man = synth.generate(s.frames_dir, cfg,
                             progress=lambda i, n, m: s.progress(i, n, m))
        s.synth_manifest = man
        s.set_job(stage="reading frame metadata", total=len(man["files"]))
        for i, path in enumerate(man["files"]):
            s.progress(i, len(man["files"]), os.path.basename(path))
            s.frames.append(ingest.scan_frame(path, i))
        _post_ingest(s)
        s.phot_cfg.gain = cfg.gain
        s.phot_cfg.read_noise = cfg.read_noise
        s.phot_cfg.saturation = cfg.saturation
        s.note(f"simulated {man['n_frames']} frames, {man['span_hours']:.2f} h, "
               f"{man['cycles']:.2f} cycles")

    return ok(run_bg(s, "simulating exposures", work))


@app.get("/api/session/{sid}/frames")
def list_frames(sid: str):
    return ok(frames_payload(get_session(sid)))


@app.post("/api/session/{sid}/times")
def set_times(sid: str, req: TimesReq):
    s = get_session(sid)
    busy_guard(s)
    if not s.frames:
        raise HTTPException(400, "Load frames first.")
    try:
        ingest.apply_manual_times(s.frames, req.start_utc, req.cadence_s, req.exptime_s)
    except Exception as exc:
        raise HTTPException(400, f"Could not parse the start time: {exc}")
    s.note(f"manual timestamps applied from {req.start_utc}, "
           f"{req.cadence_s}s cadence")
    return ok(frames_payload(s))


@app.post("/api/session/{sid}/exclude")
def exclude_frames(sid: str, indices: list[int] = Body(...), excluded: bool = Body(True)):
    s = get_session(sid)
    busy_guard(s)
    for i in indices:
        if 0 <= i < len(s.frames):
            s.frames[i].excluded = bool(excluded)
    return ok(frames_payload(s))


# ===========================================================================
# routes - preview & detection
# ===========================================================================

@app.get("/api/session/{sid}/preview")
def preview(sid: str, frame: int = 0, channel: str = "G", stretch: str = "asinh"):
    s = get_session(sid)
    if not s.frames:
        raise HTTPException(400, "Load frames first.")
    frame = max(0, min(frame, len(s.frames) - 1))
    try:
        data, _, _, _, _ = ingest.read_pixels(s.frames[frame].path, channel)
    except Exception as exc:
        raise HTTPException(400, f"Could not read frame: {exc}")
    png = phot_mod.preview_png(data, stretch)
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "no-store",
                             "X-Image-Height": str(data.shape[0]),
                             "X-Image-Width": str(data.shape[1])})


@app.post("/api/session/{sid}/detect")
def detect(sid: str, req: DetectReq):
    s = get_session(sid)
    busy_guard(s)
    if not s.frames:
        raise HTTPException(400, "Load frames first.")
    idx = max(0, min(req.frame, len(s.frames) - 1))
    try:
        data, _, _, used, sat = ingest.read_pixels(s.frames[idx].path, req.channel)
    except Exception as exc:
        raise HTTPException(400, f"Could not read frame: {exc}")

    cfg = phot_mod.PhotConfig(channel=req.channel, fwhm=req.fwhm,
                              thresh_sigma=req.thresh_sigma,
                              max_sources=req.max_sources,
                              saturation=s.phot_cfg.saturation or sat,
                              gain=s.phot_cfg.gain,
                              read_noise=s.phot_cfg.read_noise)
    srcs = phot_mod.detect_sources(data, cfg)
    measured = phot_mod.measure_fwhm(data, [(x["x"], x["y"]) for x in srcs[:10]])
    s.sources = srcs
    s.ref_index = idx
    s.phot_cfg.channel = req.channel
    s.phot_cfg.fwhm = measured or req.fwhm
    s.phot_cfg.thresh_sigma = req.thresh_sigma
    mean, median, std = phot_mod.background_stats(data)

    # Suggest the brightest unsaturated stars as comparisons, and - for a
    # synthetic field - point at the true target so the demo is one click.
    suggested_target = None
    if s.synth_manifest:
        tstars = s.synth_manifest["truth"]["stars_frame1"]
        tgt = next((t for t in tstars if t["target"]), None)
        if tgt and srcs:
            d = [np.hypot(x["x"] - tgt["x"], x["y"] - tgt["y"]) for x in srcs]
            if min(d) < 5:
                suggested_target = int(np.argmin(d))
    if suggested_target is None and srcs:
        h, w = data.shape
        d = [np.hypot(x["x"] - w / 2, x["y"] - h / 2) for x in srcs]
        suggested_target = int(np.argmin(d))
    suggested_comps = [i for i, x in enumerate(srcs)
                       if i != suggested_target and not x["saturated"]][:5]

    s.note(f"detected {len(srcs)} sources on frame {idx + 1}; FWHM {s.phot_cfg.fwhm:.2f} px")
    return ok({
        "sources": srcs, "n": len(srcs), "frame": idx,
        "shape": list(data.shape), "channel_used": used,
        "measured_fwhm": measured, "saturation": cfg.saturation,
        "background": {"mean": mean, "median": median, "std": std},
        "aperture": {"r_ap": cfg.r_ap, "r_in": cfg.r_in, "r_out": cfg.r_out},
        "suggested_target": suggested_target,
        "suggested_comps": suggested_comps,
        "gain": cfg.gain,
    })


# ===========================================================================
# routes - photometry
# ===========================================================================

@app.post("/api/session/{sid}/photometry")
def photometry(sid: str, req: PhotReq):
    s = get_session(sid)
    busy_guard(s)
    if not s.sources:
        raise HTTPException(400, "Detect stars first.")
    n = len(s.sources)
    if not (0 <= req.target < n):
        raise HTTPException(400, "Target index out of range.")
    comps = [c for c in req.comps if 0 <= c < n and c != req.target]
    if not comps:
        raise HTTPException(400,
                            "Select at least one comparison star. Differential photometry "
                            "divides the target by the comparison ensemble - that is what "
                            "removes cloud, haze and airmass changes.")

    cfg = phot_mod.PhotConfig(
        channel=req.channel, fwhm=req.fwhm, ap_factor=req.ap_factor,
        ann_in_factor=req.ann_in_factor, ann_out_factor=req.ann_out_factor,
        gain=req.gain, read_noise=req.read_noise, track=req.track,
        global_align=req.global_align, track_box=req.track_box,
        saturation=s.phot_cfg.saturation, thresh_sigma=s.phot_cfg.thresh_sigma,
    )
    s.phot_cfg = cfg
    s.target, s.comps = req.target, comps
    s.comp_mags = {int(k): (float(v) if v not in (None, "") else None)
                   for k, v in (req.comp_mags or {}).items()}
    positions = [(x["x"], x["y"]) for x in s.sources]

    def work():
        s.set_job(stage="measuring apertures", total=len(s.usable_frames))
        res = phot_mod.run_photometry(
            s.usable_frames, positions, cfg, s.ref_index,
            progress=lambda i, t, m: s.progress(i, t, m))
        s.phot = res
        s.diff = phot_mod.differential(res, req.target, comps,
                                       comp_mags=s.comp_mags or None,
                                       saturation=cfg.saturation)
        s.period_result = None
        s.distance_result = None
        s.plots = {}
        g = s.diff["good"]
        s.note(f"photometry done: {int(g.sum())}/{len(res.jd)} usable points, "
               f"scatter {np.nanstd(s.diff['dmag'][g]) * 1000:.1f} mmag")

    return ok(run_bg(s, "measuring apertures", work))


@app.get("/api/session/{sid}/photometry")
def photometry_result(sid: str):
    s = get_session(sid)
    if s.phot is None or s.diff is None:
        raise HTTPException(400, "Run photometry first.")
    res, d = s.phot, s.diff
    g = d["good"]
    ng = int(g.sum())

    # 8-bit files are a common and specific trap: they have already been
    # stretched for display, which clips the star cores flat.
    depth_note = None
    sat = s.phot_cfg.saturation
    if sat and sat <= 255 and d.get("n_rejected_saturated"):
        depth_note = (
            "These are 8-bit images (full scale 255). An 8-bit file has already had a "
            "display stretch applied, which flattens the bright cores of exactly the "
            "stars you want to measure - the pixel values are no longer proportional "
            "to the photon count, so photometry on them is not recoverable. Re-export "
            "from the original captures as FITS or 16-bit TIFF with no curve, no gamma "
            "and no auto-brightness. If these came from a DSLR, upload the RAW files "
            "(.cr2/.cr3/.nef/.arw/.dng) directly instead.")

    if ng == 0:
        return ok({"n_good": 0, "n_total": int(len(res.jd)),
                   "rejection_note": d.get("rejection_note"),
                   "bitdepth_note": depth_note,
                   "n_rejected_saturated": d.get("n_rejected_saturated"),
                   "saturated_stars": d.get("saturated_stars"),
                   "failures": res.failures[:20]})

    dm = d["dmag"][g]
    sig = d["sigma"][g]
    return ok({
        "n_total": int(len(res.jd)), "n_good": ng,
        "n_rejected_saturated": d.get("n_rejected_saturated"),
        "rejection_note": d.get("rejection_note"),
        "bitdepth_note": depth_note,
        "edge_note": res.edge_note,
        "saturated_stars": d.get("saturated_stars"),
        "failures": res.failures[:20],
        "scatter_mmag": float(np.nanstd(dm) * 1000),
        "median_sigma_mmag": float(np.nanmedian(sig) * 1000),
        "range_mmag": float((np.nanmax(dm) - np.nanmin(dm)) * 1000),
        "calibrated": d["calibrated"],
        "zeropoint": d["zeropoint"], "zeropoint_sigma": d["zeropoint_sigma"],
        "n_calibrators": d["n_calibrators"],
        "zeropoint_note": d.get("zeropoint_note"),
        "zeropoint_floored": d.get("zeropoint_floored"),
        "median_fwhm_px": float(np.nanmedian(res.fwhm)),
        "drift_px": float(np.hypot(
            res.xpos[-1, s.target] - res.xpos[0, s.target],
            res.ypos[-1, s.target] - res.ypos[0, s.target])) if len(res.jd) > 1 else 0.0,
        "stars": phot_mod.comparison_report(res, s.target, s.comps),
        "aperture": {"r_ap": s.phot_cfg.r_ap, "r_in": s.phot_cfg.r_in,
                     "r_out": s.phot_cfg.r_out, "fwhm": s.phot_cfg.fwhm},
        "target": s.target, "comps": s.comps,
    })


# ===========================================================================
# routes - period
# ===========================================================================

@app.post("/api/session/{sid}/period")
def period_analysis(sid: str, req: PeriodReq):
    s = get_session(sid)
    busy_guard(s)
    if s.diff is None:
        raise HTTPException(400, "Run photometry first.")
    g = s.diff["good"]
    if int(g.sum()) < 6:
        raise HTTPException(400, "Not enough usable photometric points for a period.")

    def work():
        s.set_job(stage="period analysis", total=6)

        # --- time system -------------------------------------------------
        ra = req.ra or s.target_info.get("ra_str")
        dec = req.dec or s.target_info.get("dec_str")
        site = None
        if req.site_lat is not None and req.site_lon is not None:
            site = {"lat": req.site_lat, "lon": req.site_lon,
                    "elev": req.site_elev or 0.0}
        else:
            for f in s.frames[:5]:
                site = ingest.observatory_from_header(f.header_preview)
                if site:
                    break
        s.progress(1, 6, "converting time system")
        t, label, delta = timing.convert_times(s.diff["jd"][g], ra, dec, site,
                                               req.time_system)
        s.time_label, s.times = label, t

        y_raw = s.diff["dmag"][g]
        dy = s.diff["sigma"][g]
        y, trend = per_mod.detrend(t, y_raw, req.detrend_order) \
            if req.detrend_order is not None and req.detrend_order >= 0 else (y_raw, None)

        # --- periodogram --------------------------------------------------
        s.progress(2, 6, "Lomb-Scargle periodogram")
        ls = per_mod.lomb_scargle(t, y, dy, req.p_min, req.p_max, req.oversample)
        cat_p = req.catalog_period if req.catalog_period is not None \
            else s.target_info.get("period_cat")

        freq_used = ls["freq_best"]
        forced = False
        if req.force_period and req.force_period > 0:
            freq_used = 1.0 / float(req.force_period)
            forced = True

        peaks = per_mod.top_peaks(ls["freq"], ls["power"], 8, span=ls["span"])
        aliases = per_mod.detect_aliases(ls["freq"], ls["power"], freq_used,
                                         ls["span"], catalog_period=cat_p)

        # --- harmonic fit -------------------------------------------------
        s.progress(3, 6, "fitting harmonics")
        fit = per_mod.fourier_fit(t, y, dy, freq_used, nharm=req.nharm)
        period = 1.0 / freq_used

        # --- uncertainties ------------------------------------------------
        s.progress(4, 6, "estimating uncertainty")
        unc = per_mod.period_uncertainty(t, fit["rms"], fit["amp_semi"], len(t), period)
        boot = {"sigma_period": float("nan"), "periods": [], "n_iter": 0}
        if req.bootstrap and req.bootstrap > 0:
            boot = per_mod.bootstrap_period(t, y, dy, freq_used,
                                            n_iter=int(req.bootstrap),
                                            nharm=min(req.nharm, 3))

        # --- PDM cross-check ----------------------------------------------
        s.progress(5, 6, "phase dispersion minimisation")
        pdm = per_mod.pdm(t, y, req.p_min, req.p_max,
                          n_periods=int(req.pdm_periods), nbins=int(req.pdm_bins))

        cycles = ls["span"] / period if period > 0 else 0.0
        cons = per_mod.consolidate_uncertainty(period, unc["sigma_period"],
                                               boot["sigma_period"],
                                               pdm["period_best"], cycles)

        # --- multi-mode ---------------------------------------------------
        s.progress(6, 6, "searching for additional modes")
        modes = per_mod.prewhiten(t, y, dy, req.p_min, req.p_max,
                                  n_modes=int(req.n_modes), nharm=2)
        ratios = per_mod.classify_mode_ratio(modes)

        # Residual scatter after removing every significant mode is the fair
        # noise estimate for a multi-mode pulsator.
        # Remove every mode detected above S/N ~ 3 before measuring the noise.
        # Leaving a real second mode in the residuals inflates the scatter and
        # makes a genuinely clean double-mode star look noisy.
        resid_rms = fit["rms"]
        if len(modes) > 1:
            yy = y.copy()
            for m in modes:
                snr = m.get("snr")
                if snr is None or not np.isfinite(snr) or snr < 3.0:
                    break
                f2 = per_mod.fourier_fit(t, yy, dy, m["freq"], nharm=2)
                yy = np.asarray(f2["resid"], float)
            resid_rms = min(resid_rms, float(np.std(yy)))

        assess = per_mod.assess(t, period, cons["sigma_period"], len(t), ls["fap"],
                                fit["amp_semi"], resid_rms,
                                comps_ok=_comps_ok(s), aliases=aliases)

        phase = per_mod.phase_fold(t, period, fit["t_max"])
        binned = per_mod.bin_phase(phase, y, 40)

        cal_fit = None
        if s.diff["calibrated"]:
            cal = s.diff["mag"][g]
            cal_fit = per_mod.fourier_fit(t, cal, dy, freq_used, nharm=req.nharm)

        s.period_result = {
            "time_label": label, "time_correction_s": delta,
            "site": site, "ra": ra, "dec": dec,
            "t": t, "y": y, "dy": dy, "y_raw": y_raw,
            "phase": phase, "binned": binned,
            "ls": ls, "peaks": peaks, "aliases": aliases,
            "fit": fit, "cal_fit": cal_fit, "pdm": pdm, "modes": modes,
            "mode_ratios": ratios, "unc": unc, "boot": boot,
            "consolidated": cons, "assess": assess,
            "period": period, "freq": freq_used, "forced": forced,
            "catalog_period": cat_p, "resid_rms": resid_rms,
            "detrend_order": req.detrend_order, "nharm": req.nharm,
            "cycles": cycles,
        }
        s.distance_result = None
        s.plots = {}
        s.note(f"period {period:.7f} d +- {cons['sigma_period']:.2e} "
               f"({assess['verdict']})")

    return ok(run_bg(s, "period analysis", work))


def _comps_ok(s: Session) -> bool:
    if s.phot is None:
        return True
    rep = phot_mod.comparison_report(s.phot, s.target, s.comps)
    for r in rep:
        if r["role"] == "comparison" and r.get("check_rms_mmag"):
            if r["check_rms_mmag"] > 50:
                return False
    return True


@app.get("/api/session/{sid}/period")
def period_result(sid: str):
    s = get_session(sid)
    r = s.period_result
    if r is None:
        raise HTTPException(400, "Run the period analysis first.")
    fit, ls = r["fit"], r["ls"]
    out = {
        "period_days": r["period"], "period_hours": r["period"] * 24.0,
        "frequency_cd": r["freq"], "forced": r["forced"],
        "sigma_period_days": r["consolidated"]["sigma_period"],
        "sigma_period_hours": r["consolidated"]["sigma_period"] * 24.0,
        "sigma_estimates": r["consolidated"]["estimates"],
        "sigma_driver": r["consolidated"]["driver"],
        "sigma_rationale": r["consolidated"]["rationale"],
        "rel_precision_pct": (r["consolidated"]["sigma_period"] / r["period"] * 100
                              if r["period"] else None),
        "power": ls["power_best"], "fap": ls["fap"],
        "fap_error": ls.get("fap_error"),
        "rayleigh_df": ls["rayleigh_df"],
        "rayleigh_period": r["unc"]["rayleigh_period"],
        "span_days": ls["span"], "span_hours": ls["span"] * 24.0,
        "cycles": r["cycles"], "n_points": ls["n"],
        "peaks": r["peaks"], "aliases": r["aliases"],
        "pdm_period": r["pdm"]["period_best"], "pdm_theta": r["pdm"]["theta_best"],
        "pdm_vs_ls_pct": (abs(r["pdm"]["period_best"] - r["period"]) / r["period"] * 100
                          if r["period"] and np.isfinite(r["pdm"]["period_best"]) else None),
        "amplitude_semi_mag": fit["amp_semi"],
        "amplitude_p2p_mag": fit["amp_peak_to_peak"],
        "harmonic_amps": fit["harmonic_amps"],
        "residual_rms_mmag": r["resid_rms"] * 1000,
        "fit_rms_mmag": fit["rms"] * 1000,
        "chi2_red": fit["chi2_red"],
        "t_max": fit["t_max"], "time_label": r["time_label"],
        "time_correction_s": r["time_correction_s"],
        "mean_dmag": fit["mag_mean_fourier"],
        "modes": r["modes"], "mode_ratios": r["mode_ratios"],
        "assess": r["assess"],
        "catalog_period": r["catalog_period"],
        "catalog_diff_pct": (abs(r["period"] - r["catalog_period"]) / r["catalog_period"] * 100
                             if r["catalog_period"] else None),
        "calibrated": s.diff["calibrated"],
        "site": r["site"],
    }
    if r["cal_fit"]:
        cf = r["cal_fit"]
        out["mean_mag_v"] = cf["mag_mean_intensity"]
        out["mean_mag_v_magavg"] = cf["mag_mean_fourier"]
        out["mag_at_max"] = cf["mag_at_max"]
        out["mag_at_min"] = cf["mag_at_min"]
        out["sigma_mean_mag_v"] = float(np.hypot(
            s.diff["zeropoint_sigma"] or 0.02, cf["rms"] / max(1, np.sqrt(cf["n"]))))
    if r["catalog_period"] and r["period"]:
        # Cycle count since the published epoch, for an O-C comparison.
        ep = s.target_info.get("epoch_cat")
        if ep:
            e = (fit["t_max"] - ep) / r["catalog_period"]
            out["epoch_cycles_elapsed"] = e
            out["oc_minutes"] = ((e - round(e)) * r["catalog_period"] * 1440.0)
    return ok(out)


# ===========================================================================
# routes - distance
# ===========================================================================

@app.post("/api/session/{sid}/distance")
def distance_calc(sid: str, req: DistReq):
    s = get_session(sid)
    r = s.period_result
    if r is None:
        raise HTTPException(400, "Measure a period first.")

    # ---- which period -------------------------------------------------
    if req.period_source == "catalog":
        p = r["catalog_period"] or s.target_info.get("period_cat")
        sp = 1e-7
        p_src = f"published catalog period ({p:.7f} d)"
        if not p:
            raise HTTPException(400, "No catalog period available.")
    elif req.period_source == "manual":
        if not req.manual_period:
            raise HTTPException(400, "Enter a period.")
        p, sp = float(req.manual_period), 0.0
        p_src = "manually entered period"
    else:
        p, sp = r["period"], r["consolidated"]["sigma_period"]
        p_src = "period measured from your photometry"

    # For a double-mode HADS the P-L relation needs the radial fundamental,
    # which is the longer of the two periods.
    fundamental_note = None
    if req.use_fundamental and r["mode_ratios"]:
        mr = r["mode_ratios"][0]
        if abs(mr["overtone_period"] - p) / p < 0.02:
            p = mr["fundamental_period"]
            fundamental_note = (
                "The measured period matched the first-overtone mode. The "
                f"fundamental ({p:.6f} d) was used instead, as the "
                "period-luminosity relation is calibrated on it.")

    # ---- which mean magnitude ------------------------------------------
    if req.mean_mag is not None:
        mv = float(req.mean_mag)
        smv = float(req.sigma_mean_mag if req.sigma_mean_mag is not None else 0.05)
        m_src = "mean apparent magnitude entered manually"
    elif r["cal_fit"]:
        cf = r["cal_fit"]
        mv = cf["mag_mean_intensity"]
        smv = float(np.hypot(s.diff["zeropoint_sigma"] or 0.02,
                             cf["rms"] / max(1.0, np.sqrt(cf["n"]))))
        m_src = (f"intensity-weighted mean of your calibrated light curve "
                 f"({s.diff['n_calibrators']} comparison stars with catalog magnitudes)")
    else:
        raise HTTPException(
            400,
            "A distance needs an apparent magnitude, and your light curve is "
            "differential only. Either enter catalog V magnitudes for one or more "
            "comparison stars (AAVSO's Variable Star Plotter or APASS will give you "
            "these) and re-run photometry, or type the star's mean apparent V "
            "magnitude directly.")

    custom = None
    if req.custom_slope is not None and req.custom_intercept is not None:
        custom = {"slope": req.custom_slope, "intercept": req.custom_intercept,
                  "slope_err": req.custom_slope_err or 0.0,
                  "intercept_err": req.custom_intercept_err or 0.0,
                  "scatter": req.custom_scatter or 0.0,
                  "label": "custom relation"}

    sol = dist_mod.solve(p, sp, mv, smv, relation=req.relation, ebv=req.ebv,
                         a_v=req.a_v, sigma_ext=req.sigma_ext, custom=custom)
    sol["period_source"] = p_src
    sol["mean_mag_source"] = m_src
    sol["fundamental_note"] = fundamental_note
    sol["properties"] = dist_mod.stellar_properties(
        sol["absolute"]["M"], sol["absolute"]["sigma_M"], p, teff=req.teff)

    if s.gaia and s.gaia.get("ok"):
        sol["gaia_comparison"] = dist_mod.compare(sol["distance"], s.gaia)
        sol["gaia"] = s.gaia

    # A quality caveat carried forward from the period step.
    v = r["assess"]["verdict"]
    if v in ("Insufficient", "Provisional"):
        sol["quality_warning"] = (
            f"The period this distance rests on was graded '{v}'. "
            + (r["assess"].get("verdict_note") or "")
            + " The distance inherits every one of those caveats.")

    s.distance_result = sol
    s.plots.pop("distance", None)
    s.plots.pop("plrelation", None)
    s.note(f"distance {sol['distance']['distance_pc']:.0f} pc "
           f"({sol['distance']['distance_ly']:.0f} ly)")
    return ok(sol)


@app.post("/api/session/{sid}/lookup/ebv")
def lookup_ebv(sid: str, req: LookupReq):
    s = get_session(sid)
    ra = req.ra or s.target_info["ra_str"]
    dec = req.dec or s.target_info["dec_str"]
    try:
        c = timing.parse_coord(ra, dec)
    except Exception as exc:
        raise HTTPException(400, f"Could not parse coordinates: {exc}")
    return ok(dist_mod.fetch_ebv(c.ra.deg, c.dec.deg))


@app.post("/api/session/{sid}/lookup/gaia")
def lookup_gaia(sid: str, req: LookupReq):
    s = get_session(sid)
    ra = req.ra or s.target_info["ra_str"]
    dec = req.dec or s.target_info["dec_str"]
    try:
        c = timing.parse_coord(ra, dec)
    except Exception as exc:
        raise HTTPException(400, f"Could not parse coordinates: {exc}")
    res = dist_mod.fetch_gaia(c.ra.deg, c.dec.deg)
    s.gaia = res
    if res.get("ok") and s.distance_result:
        s.distance_result["gaia"] = res
        s.distance_result["gaia_comparison"] = dist_mod.compare(
            s.distance_result["distance"], res)
        s.plots.pop("distance", None)
    return ok(res)


# ===========================================================================
# routes - plots
# ===========================================================================

@app.get("/api/session/{sid}/plot/{name}")
def plot(sid: str, name: str, theme: str = "dark"):
    s = get_session(sid)
    theme = "light" if theme == "light" else "dark"
    key = f"{name}:{theme}"
    if key in s.plots:
        return Response(s.plots[key], media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    r = s.period_result
    try:
        png = _render(s, r, name, theme)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Could not render '{name}': {exc}")
    if png is None:
        raise HTTPException(404, f"No plot named '{name}' is available yet.")
    s.plots[key] = png
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


def _render(s: Session, r, name, theme):
    if name == "rawcurve":
        if s.diff is None:
            raise HTTPException(400, "Run photometry first.")
        g = s.diff["good"]
        _, labels = per_mod.session_stats(s.diff["jd"][g])
        return pl.light_curve(s.diff["jd"][g], s.diff["dmag"][g], s.diff["sigma"][g],
                              theme=theme,
                              title="Differential light curve (before period analysis)",
                              sessions=labels if labels is not None and len(set(labels)) > 1 else None)

    if s.diff is not None and name == "diagnostics" and s.phot is not None:
        g = s.diff["good"]
        res = s.phot
        return pl.diagnostics(s.diff["jd"][g], res.fwhm[g], res.sky[g][:, s.target],
                              res.xpos[g][:, s.target], res.ypos[g][:, s.target],
                              s.diff["flux_comp"][g], theme=theme)

    if name == "field":
        if not s.sources:
            raise HTTPException(400, "Detect stars first.")
        shape = s.frames[s.ref_index].shape
        return pl.field_chart(shape, s.sources, s.target, s.comps, theme=theme,
                              r_ap=s.phot_cfg.r_ap)

    if r is None:
        raise HTTPException(400, "Run the period analysis first.")

    if name == "lightcurve":
        _, labels = per_mod.session_stats(r["t"])
        multi = labels is not None and len(set(labels)) > 1
        return pl.light_curve(r["t"], r["y"], r["dy"], r["t"], r["fit"]["model"],
                              theme=theme, ylabel="Delta magnitude",
                              title=f"Light curve ({r['time_label']})",
                              sessions=labels if multi else None)
    if name == "periodogram":
        ls = r["ls"]
        return pl.periodogram(ls["freq"], ls["power"], r["freq"], r["peaks"],
                              theme=theme, catalog_period=r["catalog_period"],
                              rayleigh=ls["rayleigh_df"])
    if name == "pdm":
        return pl.pdm_plot(r["pdm"]["period"], r["pdm"]["theta"],
                           r["pdm"]["period_best"], r["catalog_period"], theme=theme)
    if name == "folded":
        return pl.folded(r["phase"], r["y"], r["dy"], r["binned"],
                         r["fit"]["curve_phase"], r["fit"]["curve_mag"],
                         r["period"], theme=theme)
    if name == "foldedcal" and r["cal_fit"]:
        g = s.diff["good"]
        cal = s.diff["mag"][g]
        ph = per_mod.phase_fold(r["t"], r["period"], r["cal_fit"]["t_max"])
        return pl.folded(ph, cal, r["dy"], per_mod.bin_phase(ph, cal, 40),
                         r["cal_fit"]["curve_phase"], r["cal_fit"]["curve_mag"],
                         r["period"], theme=theme, ylabel="Apparent V magnitude",
                         title=f"Calibrated V, folded at P = {r['period']:.6f} d")
    if name == "bootstrap" and r["boot"]["periods"]:
        return pl.bootstrap_hist(r["boot"]["periods"], r["period"], theme=theme)
    if name == "distance" and s.distance_result:
        return pl.distance_summary(s.distance_result, s.gaia, theme=theme)
    if name == "plrelation" and s.distance_result:
        return pl.pl_relation(s.distance_result, theme=theme)
    return None


# ===========================================================================
# routes - export
# ===========================================================================

@app.get("/api/session/{sid}/export/lightcurve.csv")
def export_csv(sid: str):
    s = get_session(sid)
    if s.diff is None:
        raise HTTPException(400, "Run photometry first.")
    r = s.period_result
    g = s.diff["good"]
    res = s.phot

    lines = []
    lines.append("# VarStar Lab light curve export")
    lines.append(f"# target star index {s.target}, comparison stars {s.comps}")
    lines.append(f"# aperture r={s.phot_cfg.r_ap:.2f} px, sky annulus "
                 f"{s.phot_cfg.r_in:.2f}-{s.phot_cfg.r_out:.2f} px, "
                 f"gain={s.phot_cfg.gain} e-/ADU")
    if r:
        lines.append(f"# time system: {r['time_label']}")
        lines.append(f"# period {r['period']:.8f} d +- "
                     f"{r['consolidated']['sigma_period']:.2e} d")
    cols = ["time", "dmag", "dmag_err"]
    if s.diff["calibrated"]:
        cols.append("V_mag")
    if r:
        cols += ["phase", "model_dmag"]
    cols += ["flux_target", "flux_comp_sum", "x_target", "y_target", "fwhm_px",
             "sky_target", "filename"]
    lines.append(",".join(cols))

    t = r["t"] if r else s.diff["jd"][g]
    idx = np.where(g)[0]
    model = r["fit"]["model"] if r else None
    for k, i in enumerate(idx):
        row = [f"{t[k]:.8f}", f"{s.diff['dmag'][i]:.6f}", f"{s.diff['sigma'][i]:.6f}"]
        if s.diff["calibrated"]:
            row.append(f"{s.diff['mag'][i]:.6f}")
        if r:
            row.append(f"{r['phase'][k]:.6f}")
            row.append(f"{model[k]:.6f}")
        row += [f"{res.flux[i, s.target]:.3f}", f"{s.diff['flux_comp'][i]:.3f}",
                f"{res.xpos[i, s.target]:.3f}", f"{res.ypos[i, s.target]:.3f}",
                f"{res.fwhm[i]:.3f}" if np.isfinite(res.fwhm[i]) else "",
                f"{res.sky[i, s.target]:.3f}", res.filenames[i]]
        lines.append(",".join(row))

    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/csv",
                             headers={"Content-Disposition":
                                      'attachment; filename="lightcurve.csv"'})


def _report_dict(s: Session):
    r, d = s.period_result, s.distance_result
    rep = {
        "app": "VarStar Lab", "version": "1.0.0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": s.target_info,
        "data": {
            "n_frames_loaded": len(s.frames),
            "n_frames_measured": int(len(s.phot.jd)) if s.phot else 0,
            "synthetic": s.synth_manifest is not None,
            "synthetic_truth": ({k: v for k, v in s.synth_manifest["truth"].items()
                                 if k not in ("rows", "stars", "stars_frame1")}
                                if s.synth_manifest else None),
        },
        "photometry": {
            "target_star": s.target, "comparison_stars": s.comps,
            "comparison_catalog_mags": s.comp_mags,
            "channel": s.phot_cfg.channel,
            "fwhm_px": s.phot_cfg.fwhm,
            "aperture_radius_px": s.phot_cfg.r_ap,
            "annulus_px": [s.phot_cfg.r_in, s.phot_cfg.r_out],
            "gain_e_per_adu": s.phot_cfg.gain,
            "read_noise_e": s.phot_cfg.read_noise,
            "tracking": s.phot_cfg.track,
        },
    }
    if s.diff is not None:
        g = s.diff["good"]
        rep["photometry"].update({
            "n_good_points": int(g.sum()),
            "scatter_mmag": float(np.nanstd(s.diff["dmag"][g]) * 1000),
            "median_sigma_mmag": float(np.nanmedian(s.diff["sigma"][g]) * 1000),
            "calibrated": s.diff["calibrated"],
            "zeropoint": s.diff["zeropoint"],
            "zeropoint_sigma": s.diff["zeropoint_sigma"],
            "stars": phot_mod.comparison_report(s.phot, s.target, s.comps),
        })
    if r:
        rep["period"] = {
            "time_system": r["time_label"],
            "time_correction_s": r["time_correction_s"],
            "observing_site": r["site"],
            "period_days": r["period"], "period_hours": r["period"] * 24,
            "sigma_period_days": r["consolidated"]["sigma_period"],
            "sigma_estimates": r["consolidated"]["estimates"],
            "sigma_rationale": r["consolidated"]["rationale"],
            "frequency_cd": r["freq"],
            "lomb_scargle_power": r["ls"]["power_best"],
            "false_alarm_probability": r["ls"]["fap"],
            "pdm_period": r["pdm"]["period_best"],
            "pdm_theta": r["pdm"]["theta_best"],
            "rayleigh_resolution_cd": r["ls"]["rayleigh_df"],
            "span_hours": r["ls"]["span"] * 24,
            "cycles_covered": r["cycles"],
            "n_points": r["ls"]["n"],
            "amplitude_semi_mag": r["fit"]["amp_semi"],
            "amplitude_peak_to_peak_mag": r["fit"]["amp_peak_to_peak"],
            "epoch_of_maximum": r["fit"]["t_max"],
            "residual_rms_mmag": r["resid_rms"] * 1000,
            "harmonic_amplitudes": r["fit"]["harmonic_amps"],
            "top_peaks": r["peaks"],
            "aliases": r["aliases"],
            "modes": r["modes"], "mode_ratios": r["mode_ratios"],
            "assessment": r["assess"],
            "catalog_period": r["catalog_period"],
        }
        if r["cal_fit"]:
            rep["period"]["mean_apparent_V"] = r["cal_fit"]["mag_mean_intensity"]
    if d:
        rep["distance"] = d
    rep["method_notes"] = [
        "Differential aperture photometry against an ensemble of comparison stars; "
        "each frame re-registered by cross-correlation plus per-star centroiding, "
        "because the proposal's observing plan turns auto-guiding off.",
        "Sky subtracted as the sigma-clipped median of an annulus around each star; "
        "flux errors from the CCD equation with the annulus scatter as the sky term.",
        "Period from a Lomb-Scargle periodogram, cross-checked against phase "
        "dispersion minimisation, which assumes nothing about the curve shape.",
        "Quoted period uncertainty is the largest of the analytic (Montgomery & "
        "O'Donoghue 1999), residual-bootstrap, and LS-vs-PDM disagreement estimates.",
        "Distance from the delta Scuti period-luminosity relation: M_V from the "
        "period, then d = 10^((<V> - M_V - A_V)/5 + 1) parsecs.",
    ]
    return rep


@app.get("/api/session/{sid}/export/report.json")
def export_json(sid: str):
    s = get_session(sid)
    body = json.dumps(jsonable(_report_dict(s)), indent=2)
    return PlainTextResponse(body, media_type="application/json",
                             headers={"Content-Disposition":
                                      'attachment; filename="varstar_report.json"'})


@app.get("/api/session/{sid}/export/bundle.zip")
def export_zip(sid: str, theme: str = "light"):
    s = get_session(sid)
    if s.diff is None:
        raise HTTPException(400, "Run photometry first.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.json", json.dumps(jsonable(_report_dict(s)), indent=2))
        z.writestr("lightcurve.csv", export_csv(sid).body.decode())
        for name in ("field", "rawcurve", "lightcurve", "periodogram", "pdm",
                     "folded", "foldedcal", "diagnostics", "bootstrap",
                     "distance", "plrelation"):
            try:
                png = _render(s, s.period_result, name, theme)
            except Exception:
                png = None
            if png:
                z.writestr(f"plots/{name}.png", png)
        z.writestr("README.txt",
                   "VarStar Lab export\n"
                   "==================\n\n"
                   "report.json     - every number, with the method notes\n"
                   "lightcurve.csv  - one row per exposure\n"
                   "plots/          - figures rendered on a white background\n")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition":
                                      'attachment; filename="varstar_lab_export.zip"'})


# ===========================================================================
# static
# ===========================================================================

@app.get("/api/health")
def health():
    with _LOCK:
        n = len(SESSIONS)
    return ok({"ok": True, "sessions": n, "port": PORT,
               "max_frames": MAX_FRAMES})


@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join(STATIC, "index.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>VarStar Lab</h1><p>static/index.html missing</p>",
                            status_code=500)
    return FileResponse(path)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    print(f"VarStar Lab -> http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
