#!/usr/bin/env python3
"""Drive the whole app through its HTTP API, the way the browser does.

Usage: python3 apitest.py [preset] [base_url]
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:12113"


def req(method, path, body=None, raw=False):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                              headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            blob = resp.read()
            if raw:
                return resp.status, blob, dict(resp.headers)
            return json.loads(blob) if blob else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:400]
        raise SystemExit(f"HTTP {e.code} on {method} {path}: {detail}")


def wait(sid, label):
    t0 = time.time()
    last = ""
    while True:
        time.sleep(0.5)
        j = req("GET", f"/api/session/{sid}/job")
        line = f"{j.get('stage')} {j.get('current')}/{j.get('total')} {j.get('message')}"
        if line != last:
            print(f"      {line[:96]}", flush=True)
            last = line
        if not j["running"]:
            if j.get("error"):
                raise SystemExit(f"JOB FAILED ({label}): {j['error']}\n"
                                 f"{j.get('traceback', '')}")
            print(f"      -> {label} done in {time.time() - t0:.1f}s")
            return j


def main(preset="single_night"):
    print(f"### preset = {preset}   base = {BASE}")
    print(f"[health] {req('GET', '/api/health')}")

    s = req("POST", "/api/session")
    sid = s["session_id"]
    print(f"[1] session {sid}")
    print(f"    presets:   {list(s['presets'])}")
    print(f"    relations: {list(s['relations'])}")

    print("[2] generating synthetic run")
    req("POST", f"/api/session/{sid}/demo", {"preset": preset})
    wait(sid, "demo")
    f = req("GET", f"/api/session/{sid}/frames")
    print(f"    {f['n_frames']} frames, {f['span_hours']:.2f} h, "
          f"{f['n_sessions']} session(s), time src = {f['frames'][0]['time_source']!r}")
    truth = f["synthetic_truth"]
    print(f"    TRUTH: P={truth['period']} p2p={truth['amplitude_p2p']:.3f} "
          f"meanV={truth['mean_mag']:.3f} overtone={truth['period_overtone']}")

    print("[3] preview + detection")
    st, png, hdrs = req("GET", f"/api/session/{sid}/preview?frame=0", raw=True)
    print(f"    preview PNG {len(png)} bytes, "
          f"{hdrs.get('X-Image-Width')}x{hdrs.get('X-Image-Height')}")
    d = req("POST", f"/api/session/{sid}/detect",
            {"frame": 0, "fwhm": 4.0, "thresh_sigma": 5.0, "channel": "G"})
    print(f"    {d['n']} sources; FWHM {d['measured_fwhm']:.2f} px; "
          f"sky {d['background']['median']:.1f} ADU; sat={d['saturation']}")
    print(f"    suggested target={d['suggested_target']} comps={d['suggested_comps']}")
    print(f"    saturated sources: {sum(1 for x in d['sources'] if x['saturated'])}")

    tgt, comps, srcs = d["suggested_target"], d["suggested_comps"], d["sources"]
    assert tgt is not None and comps, "no suggestion produced"

    # Give the brightest comparisons the simulator's true magnitudes, so the
    # absolute-calibration path is exercised end to end.
    known = sorted(truth["comp_mags"].values())
    comp_mags = {}
    for i, ci in enumerate(sorted(comps, key=lambda i: -srcs[i]["flux"])):
        if i < len(known):
            comp_mags[str(ci)] = known[i]
    print(f"    catalog mags supplied: {comp_mags}")

    print("[4] photometry")
    req("POST", f"/api/session/{sid}/photometry", {
        "target": tgt, "comps": comps, "comp_mags": comp_mags,
        "channel": "G", "fwhm": d["measured_fwhm"] or 4.0,
        "ap_factor": 1.5, "ann_in_factor": 3.0, "ann_out_factor": 5.0,
        "gain": d["gain"], "track": True, "global_align": True,
    })
    wait(sid, "photometry")
    p = req("GET", f"/api/session/{sid}/photometry")
    print(f"    {p['n_good']}/{p['n_total']} points, scatter {p['scatter_mmag']:.1f} mmag, "
          f"sigma {p['median_sigma_mmag']:.2f} mmag")
    print(f"    calibrated={p['calibrated']} zp={p['zeropoint']} "
          f"+-{p['zeropoint_sigma']} from {p['n_calibrators']}")
    print(f"    drift {p['drift_px']:.1f} px, FWHM {p['median_fwhm_px']:.2f} px")
    if p.get("rejection_note"):
        print(f"    NOTE: {p['rejection_note']}")
    for st_ in p["stars"]:
        if st_["role"] != "unused":
            chk = st_["check_rms_mmag"]
            print(f"      star {st_['star']:2d} {st_['role']:11s} "
                  f"S/N={st_['median_snr']:8.1f} check="
                  f"{'None' if chk is None else round(chk, 1)}")

    print("[5] period analysis")
    req("POST", f"/api/session/{sid}/period", {
        "p_min": 0.02, "p_max": 0.30, "nharm": 4, "bootstrap": 120,
        "time_system": "bjd_tdb",
    })
    wait(sid, "period")
    r = req("GET", f"/api/session/{sid}/period")
    err = (r["period_days"] - truth["period"]) / truth["period"] * 100
    print(f"    P = {r['period_days']:.7f} d ({r['period_hours']:.5f} h)")
    print(f"    TRUE ERROR = {err:+.4f}%   quoted sigma = "
          f"{r['rel_precision_pct']:.3f}%  driver={r['sigma_driver']}")
    print(f"    amp p2p {r['amplitude_p2p_mag']:.4f} (truth {truth['amplitude_p2p']:.4f})")
    fapv = r['fap']
    print(f"    FAP {'None' if fapv is None else format(fapv, '.2e')}"
          f"  err={r.get('fap_error')}  cycles {r['cycles']:.2f}  "
          f"time={r['time_label']} ({r['time_correction_s']:.0f}s)")
    pv=r['pdm_vs_ls_pct']
    print(f"    PDM {r['pdm_period']:.7f} "
          f"({'None' if pv is None else format(pv,'.3f')}% from LS)")
    a = r["assess"]
    print(f"    verdict: {a['verdict']} - {a['verdict_note'][:76]}")
    print(f"    longest session {a['longest_session_hours']:.2f} h = "
          f"{a['cycles_in_longest_session']:.2f} cycles")
    al = r["aliases"]
    print(f"    aliases: ambiguous={al['ambiguous']} n={al['n_candidates']}")
    for c in al["candidates"][:5]:
        print(f"      P={c['period']:.7f} -{c['rel_deficit'] * 100:5.2f}% "
              f"{c['relation'][:34]:34s} cat={c.get('matches_catalog')}")
    for m in r["modes"]:
        print(f"      mode{m['mode']}: P={m['period']:.6f} amp={m['amp_mmag']:.1f} mmag "
              f"S/N={m['snr']:.1f} sig={m['significant']}")
    if r.get("mean_mag_v") is not None:
        print(f"    <V> = {r['mean_mag_v']:.4f} +- {r['sigma_mean_mag_v']:.4f} "
              f"(truth {truth['mean_mag']:.4f}, "
              f"err {r['mean_mag_v'] - truth['mean_mag']:+.4f})")
    for fl in a["flags"][:3]:
        print(f"      [{fl['level']:8s}] {fl['text'][:116]}")

    print("[6] distance")
    dist = req("POST", f"/api/session/{sid}/distance", {
        "relation": "ziaali2019", "period_source": "measured",
        "ebv": 0.09, "teff": 7400,
    })
    dd = dist["distance"]
    print(f"    M_V = {dist['absolute']['M']:.3f} +- {dist['absolute']['sigma_M']:.3f}")
    print(f"    mu_0 = {dd['mu_0']:.3f} +- {dd['sigma_mu']:.3f}")
    print(f"    d = {dd['distance_pc']:.1f} pc [{dd['distance_pc_lo']:.0f},"
          f"{dd['distance_pc_hi']:.0f}] = {dd['distance_ly']:.0f} ly "
          f"({dd['error_budget']['relative_distance_pct']:.1f}%)")
    pr = dist["properties"]
    print(f"    L={pr['luminosity_lsun']:.1f} Lsun R={pr['radius_rsun']:.2f} Rsun "
          f"M={pr['mass_msun']:.2f} Msun")
    if dist.get("quality_warning"):
        print(f"    WARN: {dist['quality_warning'][:126]}")

    print("[7] plots")
    for n in ["rawcurve", "diagnostics", "field", "lightcurve", "periodogram",
              "pdm", "folded", "foldedcal", "bootstrap", "distance", "plrelation"]:
        try:
            st_, blob, _ = req("GET", f"/api/session/{sid}/plot/{n}?theme=dark", raw=True)
            print(f"    {'ok ' if blob[:4] == b'\x89PNG' else 'BAD'} {n:12s} {len(blob):7d} bytes")
        except SystemExit as e:
            print(f"    -- {n:12s} {str(e)[:76]}")

    print("[8] exports")
    st_, csv, _ = req("GET", f"/api/session/{sid}/export/lightcurve.csv", raw=True)
    lines = csv.decode().splitlines()
    print(f"    csv: {len(lines)} lines")
    print(f"      hdr: {lines[5][:96] if len(lines) > 6 else ''}")
    print(f"      row: {lines[-1][:96]}")
    st_, js, _ = req("GET", f"/api/session/{sid}/export/report.json", raw=True)
    print(f"    report.json keys: {list(json.loads(js))}")
    st_, zp, _ = req("GET", f"/api/session/{sid}/export/bundle.zip", raw=True)
    print(f"    bundle.zip: {len(zp)} bytes, valid={zp[:2] == b'PK'}")

    print("[9] error handling")
    checks = [("POST", f"/api/session/{sid}/photometry", {"target": 0, "comps": []}),
              ("POST", "/api/session/deadbeef00/period", {}),
              ("GET", f"/api/session/{sid}/plot/nosuchplot", None)]
    for method, path, body in checks:
        try:
            req(method, path, body)
            print(f"    !! expected a rejection for {path}")
        except SystemExit as e:
            print(f"    ok rejected: {str(e)[:104]}")

    print(f"\n### period error {err:+.4f}%  |  distance {dd['distance_pc']:.0f} pc"
          f"  |  verdict {a['verdict']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2:
        BASE = sys.argv[2]
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "single_night"))
