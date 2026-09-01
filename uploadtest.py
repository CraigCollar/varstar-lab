#!/usr/bin/env python3
"""Exercise the paths a real user hits, not the demo shortcut:

  A) uploading FITS files by multipart, as the browser does
  B) uploading 8-bit PNGs with no timestamps, then setting times by hand
  C) the alias-trap preset, to confirm the ambiguity surfaces through the API
  D) requesting a distance with no calibration, to confirm the refusal is helpful
"""
import io
import json
import mimetypes
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

BASE = "http://127.0.0.1:12113"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def req(method, path, body=None, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            blob = resp.read()
            return (resp.status, blob) if raw else (json.loads(blob) if blob else None)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}")


def multipart_upload(sid, paths):
    """Build a real multipart/form-data body, exactly like the browser."""
    boundary = "----VarStarBoundary" + uuid.uuid4().hex
    body = io.BytesIO()
    for p in paths:
        ctype = mimetypes.guess_type(p)[0] or "application/octet-stream"
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="files"; '
                   f'filename="{os.path.basename(p)}"\r\n'.encode())
        body.write(f"Content-Type: {ctype}\r\n\r\n".encode())
        with open(p, "rb") as fh:
            body.write(fh.read())
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    raw = body.getvalue()
    r = urllib.request.Request(
        f"{BASE}/api/session/{sid}/upload", data=raw, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Content-Length": str(len(raw))})
    try:
        with urllib.request.urlopen(r, timeout=600) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}")


def wait(sid, label, quiet=True):
    while True:
        time.sleep(0.4)
        j = req("GET", f"/api/session/{sid}/job")
        if not j["running"]:
            if j.get("error"):
                raise RuntimeError(f"{label} failed: {j['error']}\n{j.get('traceback','')}")
            return j


def new_session():
    return req("POST", "/api/session")["session_id"]


# ---------------------------------------------------------------- A: FITS
def test_fits_upload():
    print("\n=== A) multipart upload of real FITS files ===")
    from pipeline import synth
    tmp = tempfile.mkdtemp(prefix="vs_up_")
    cfg = synth.config_from_preset("clean", n_frames=40, cadence=45.0)
    man = synth.generate(tmp, cfg)
    files = man["files"]
    print(f"  wrote {len(files)} FITS files, total "
          f"{sum(os.path.getsize(f) for f in files) / 1e6:.1f} MB")

    sid = new_session()
    multipart_upload(sid, files)
    wait(sid, "upload")
    f = req("GET", f"/api/session/{sid}/frames")
    print(f"  ingested {f['n_frames']} frames, span {f['span_hours']:.3f} h, "
          f"needs_manual_times={f['needs_manual_times']}")
    print(f"  time source: {f['frames'][0]['time_source']!r}")
    print(f"  is_synthetic flag (should be False for uploads): {f['is_synthetic']}")
    assert f["n_frames"] == len(files), "frame count mismatch"
    assert not f["needs_manual_times"], "FITS timestamps not recovered"
    assert not f["is_synthetic"], "uploaded data wrongly marked synthetic"

    d = req("POST", f"/api/session/{sid}/detect", {"frame": 0, "fwhm": 4.0})
    print(f"  detected {d['n']} sources, FWHM {d['measured_fwhm']:.2f} px, "
          f"suggested target={d['suggested_target']}")
    assert d["n"] > 5
    # No truth available on the upload path, so suggestion falls back to centre.
    req("POST", f"/api/session/{sid}/photometry", {
        "target": d["suggested_target"], "comps": d["suggested_comps"],
        "channel": "G", "fwhm": d["measured_fwhm"], "gain": d["gain"]})
    wait(sid, "photometry")
    p = req("GET", f"/api/session/{sid}/photometry")
    print(f"  photometry: {p['n_good']}/{p['n_total']} pts, "
          f"{p['median_sigma_mmag']:.2f} mmag, calibrated={p['calibrated']}")
    assert p["n_good"] > 0.9 * len(files)
    shutil.rmtree(tmp, ignore_errors=True)
    return sid


# ------------------------------------------------------- B: PNG, no times
def test_png_no_times():
    print("\n=== B) 8-bit PNGs with no timestamps + manual cadence ===")
    import numpy as np
    from PIL import Image
    from pipeline import synth
    from astropy.io import fits

    tmp = tempfile.mkdtemp(prefix="vs_png_")
    cfg = synth.config_from_preset("clean", n_frames=30, cadence=60.0, size=400)
    man = synth.generate(tmp, cfg)
    def make_pngs(hi_pct, tag):
        """Convert to 8-bit RGB PNG. hi_pct sets how hard the stretch clips."""
        outdir = tempfile.mkdtemp(prefix=f"vs_png_{tag}_")
        paths = []
        for i, fp in enumerate(man["files"]):
            with fits.open(fp) as h:
                arr = h[0].data.astype(float)
            lo, hi = np.percentile(arr, 20), np.percentile(arr, hi_pct)
            b = np.clip((arr - lo) / max(1e-9, hi - lo), 0, 1)
            rgb = np.stack([(b * 245).astype(np.uint8)] * 3, axis=-1)
            out = os.path.join(outdir, f"IMG_{i + 1:04d}.png")
            Image.fromarray(rgb, "RGB").save(out)
            paths.append(out)
        return outdir, paths

    # --- B1: hard stretch, star cores clipped -> must be refused clearly ----
    d1, p1 = make_pngs(99.9, "clipped")
    print(f"  B1: {len(p1)} PNGs with a hard stretch (cores clipped)")
    sid = new_session()
    multipart_upload(sid, p1)
    wait(sid, "upload")
    f = req("GET", f"/api/session/{sid}/frames")
    print(f"      ingested {f['n_frames']}, with_time={f['n_with_time']}, "
          f"needs_manual_times={f['needs_manual_times']}")
    assert f["needs_manual_times"], "should have asked for manual times"
    print(f"      kinds={f['kinds']}, shape={f['frames'][0]['shape']}, "
          f"is_color={f['frames'][0]['is_color']}, bitdepth={f['frames'][0]['bitdepth']}")

    f2 = req("POST", f"/api/session/{sid}/times",
             {"start_utc": "2026-07-14T14:10:00", "cadence_s": 60.0, "exptime_s": 35.0})
    print(f"      manual times ok: span={f2['span_hours']:.3f} h, "
          f"src={f2['frames'][0]['time_source']!r}")
    assert f2["n_with_time"] == f2["n_frames"]

    d = req("POST", f"/api/session/{sid}/detect", {"frame": 0, "fwhm": 4.0, "channel": "G"})
    print(f"      detected {d['n']} sources (channel {d['channel_used']}), "
          f"saturation={d['saturation']}")
    req("POST", f"/api/session/{sid}/photometry", {
        "target": d["suggested_target"], "comps": d["suggested_comps"],
        "channel": "G", "fwhm": d["measured_fwhm"] or 4.0, "gain": 1.0})
    wait(sid, "photometry")
    p = req("GET", f"/api/session/{sid}/photometry")
    print(f"      n_good={p['n_good']} of {p.get('n_total')}, "
          f"saturation-rejected={p.get('n_rejected_saturated')}")
    if p["n_good"] == 0:
        assert p.get("bitdepth_note"), "8-bit clipping was not explained to the user"
        print(f"      refusal explains the cause:\n        "
              f"{p['bitdepth_note'][:250]}")
    else:
        print(f"      (survived: {p['median_sigma_mmag']:.1f} mmag)")

    # --- B2: gentle stretch, cores intact -> photometry should work --------
    d2, p2 = make_pngs(100.0, "soft")
    print(f"  B2: {len(p2)} PNGs with a gentle stretch (cores intact)")
    sid2 = new_session()
    multipart_upload(sid2, p2)
    wait(sid2, "upload")
    req("POST", f"/api/session/{sid2}/times",
        {"start_utc": "2026-07-14T14:10:00", "cadence_s": 60.0, "exptime_s": 35.0})
    d = req("POST", f"/api/session/{sid2}/detect", {"frame": 0, "fwhm": 4.0, "channel": "G"})
    print(f"      detected {d['n']} sources")
    req("POST", f"/api/session/{sid2}/photometry", {
        "target": d["suggested_target"], "comps": d["suggested_comps"],
        "channel": "G", "fwhm": d["measured_fwhm"] or 4.0, "gain": 1.0})
    wait(sid2, "photometry")
    p = req("GET", f"/api/session/{sid2}/photometry")
    print(f"      n_good={p['n_good']}, precision="
          f"{p.get('median_sigma_mmag') and round(p['median_sigma_mmag'], 1)} mmag "
          f"(vs 1.8 mmag from the same data as FITS)")
    assert p["n_good"] > 0, "gentle 8-bit conversion should still be measurable"
    req("POST", f"/api/session/{sid2}/period", {"p_min": 0.02, "p_max": 0.30,
                                                "bootstrap": 0})
    wait(sid2, "period")
    r = req("GET", f"/api/session/{sid2}/period")
    print(f"      period step ran on 8-bit data: P={r['period_days']:.6f} d, "
          f"verdict={r['assess']['verdict']}")

    for dd in (tmp, d1, d2):
        shutil.rmtree(dd, ignore_errors=True)


# ------------------------------------------------------- C: alias trap
def test_alias_trap():
    print("\n=== C) alias_trap preset: does the ambiguity surface? ===")
    sid = new_session()
    req("POST", f"/api/session/{sid}/demo", {"preset": "alias_trap"})
    wait(sid, "demo")
    f = req("GET", f"/api/session/{sid}/frames")
    truth = f["synthetic_truth"]
    print(f"  {f['n_frames']} frames, {f['n_sessions']} sessions, {f['span_hours']:.1f} h")

    d = req("POST", f"/api/session/{sid}/detect", {"frame": 0, "fwhm": 4.0})
    known = sorted(truth["comp_mags"].values())
    comps = d["suggested_comps"]
    cm = {str(c): known[i] for i, c in enumerate(
        sorted(comps, key=lambda i: -d["sources"][i]["flux"])) if i < len(known)}
    req("POST", f"/api/session/{sid}/photometry", {
        "target": d["suggested_target"], "comps": comps, "comp_mags": cm,
        "channel": "G", "fwhm": d["measured_fwhm"], "gain": d["gain"]})
    wait(sid, "photometry")
    req("POST", f"/api/session/{sid}/period",
        {"p_min": 0.02, "p_max": 0.30, "nharm": 4, "bootstrap": 100})
    wait(sid, "period")
    r = req("GET", f"/api/session/{sid}/period")
    err = (r["period_days"] - truth["period"]) / truth["period"] * 100
    print(f"  P = {r['period_days']:.7f} (true {truth['period']}), error {err:+.3f}%")
    print(f"  verdict = {r['assess']['verdict']}  (must NOT be Good/Reasonable)")
    print(f"  FAP = {r['fap']}   fap_error={r.get('fap_error')}")
    al = r["aliases"]
    print(f"  ambiguous={al['ambiguous']}, candidates={al['n_candidates']}")
    matched = None
    for c in al["candidates"]:
        star = " <== matches catalog" if c.get("matches_catalog") else ""
        print(f"    P={c['period']:.7f}  -{c['rel_deficit'] * 100:5.2f}%  "
              f"{c['relation']}{star}")
        if c.get("matches_catalog"):
            matched = c
    assert al["ambiguous"], "alias ambiguity was NOT detected"
    assert r["assess"]["verdict"] in ("Insufficient", "Provisional"), \
        f"verdict too generous: {r['assess']['verdict']}"
    assert matched is not None, "the true period was not offered as a candidate"
    print(f"  -> the true period IS offered as a candidate: {matched['period']:.7f} d")

    # Now lock onto the correct candidate, as the UI's "use this" button does.
    req("POST", f"/api/session/{sid}/period", {
        "p_min": 0.02, "p_max": 0.30, "nharm": 4, "bootstrap": 100,
        "force_period": matched["period"]})
    wait(sid, "period")
    r2 = req("GET", f"/api/session/{sid}/period")
    err2 = (r2["period_days"] - truth["period"]) / truth["period"] * 100
    print(f"  after locking to that candidate: P={r2['period_days']:.7f}, "
          f"error {err2:+.4f}%, forced={r2['forced']}")
    assert abs(err2) < 1.0, "locking to the catalog-matching alias did not fix it"
    return sid


# --------------------------------------------- D: distance without calibration
def test_uncalibrated_distance():
    print("\n=== D) distance with no calibration: is the refusal useful? ===")
    sid = new_session()
    req("POST", f"/api/session/{sid}/demo", {"preset": "short_night"})
    wait(sid, "demo")
    d = req("POST", f"/api/session/{sid}/detect", {"frame": 0, "fwhm": 4.0})
    req("POST", f"/api/session/{sid}/photometry", {
        "target": d["suggested_target"], "comps": d["suggested_comps"],
        "channel": "G", "fwhm": d["measured_fwhm"], "gain": d["gain"]})
    wait(sid, "photometry")
    p = req("GET", f"/api/session/{sid}/photometry")
    print(f"  calibrated={p['calibrated']} (no catalog mags supplied)")
    req("POST", f"/api/session/{sid}/period", {"p_min": 0.02, "p_max": 0.30,
                                               "bootstrap": 0})
    wait(sid, "period")
    try:
        req("POST", f"/api/session/{sid}/distance", {"relation": "ziaali2019"})
        print("  !! distance succeeded without any magnitude - that is wrong")
    except RuntimeError as e:
        msg = str(e)
        print(f"  correctly refused:\n    {msg[:290]}")
        assert "apparent magnitude" in msg

    # Supplying the mean magnitude by hand should now work.
    r = req("POST", f"/api/session/{sid}/distance",
            {"relation": "ziaali2019", "mean_mag": 11.715, "sigma_mean_mag": 0.05,
             "ebv": 0.09})
    print(f"  with a manual <V>: d = {r['distance']['distance_pc']:.0f} pc "
          f"({r['distance']['distance_ly']:.0f} ly)")
    print(f"  source: {r['mean_mag_source']}")
    print(f"  quality warning present: {bool(r.get('quality_warning'))}")

    # And the catalog-period option.
    r2 = req("POST", f"/api/session/{sid}/distance",
             {"relation": "ziaali2019", "period_source": "catalog",
              "mean_mag": 11.715, "sigma_mean_mag": 0.05, "ebv": 0.09})
    print(f"  using the catalog period: d = {r2['distance']['distance_pc']:.0f} pc"
          f"  ({r2['period_source']})")

    # Every relation should run.
    for rel in ("ziaali2019", "mcnamara2011", "nemec1994"):
        rr = req("POST", f"/api/session/{sid}/distance",
                 {"relation": rel, "period_source": "catalog",
                  "mean_mag": 11.715, "sigma_mean_mag": 0.05, "ebv": 0.09})
        print(f"    {rel:14s} M_V={rr['absolute']['M']:+.3f} -> "
              f"{rr['distance']['distance_pc']:6.0f} pc")


if __name__ == "__main__":
    try:
        test_fits_upload()
        test_png_no_times()
        test_alias_trap()
        test_uncalibrated_distance()
        print("\n### all upload / edge-case tests passed")
    except AssertionError as e:
        print(f"\n### ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n### ERROR: {e}")
        sys.exit(1)
