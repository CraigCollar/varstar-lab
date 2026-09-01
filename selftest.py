#!/usr/bin/env python3
"""End-to-end check: simulate a run, reduce it, and see whether the pipeline
recovers the period and mean magnitude that were injected.

Usage:  python3 selftest.py [preset]     (single_night | short_night | three_nights | clean)
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import distance as dist_mod
from pipeline import ingest, period as per_mod, photometry as phot_mod, synth, timing


def main(preset="clean"):
    t_start = time.time()
    workdir = tempfile.mkdtemp(prefix="varstar_selftest_")
    print(f"# preset: {preset}\n# workdir: {workdir}\n")

    # ---------------------------------------------------------------- simulate
    cfg = synth.config_from_preset(preset)
    man = synth.generate(workdir, cfg)
    truth = man["truth"]
    print(f"[1] generated {man['n_frames']} frames, span {man['span_hours']:.2f} h "
          f"= {man['cycles']:.2f} cycles")
    print(f"    injected P0 = {truth['period']:.7f} d, "
          f"overtone = {truth['period_overtone']}, "
          f"mean V = {truth['mean_mag']:.3f}, p2p = {truth['amplitude_p2p']:.3f} mag")

    # ------------------------------------------------------------------ ingest
    frames = [ingest.scan_frame(p, i) for i, p in enumerate(man["files"])]
    bad = [f for f in frames if f.excluded]
    assert not bad, f"unreadable frames: {bad[:2]}"
    jds = np.array([f.jd for f in frames])
    assert np.all(np.isfinite(jds)), "timestamps missing"
    print(f"[2] ingested; time source = {frames[0].time_source!r}; "
          f"exptime = {frames[0].exptime}s; shape = {frames[0].shape}")

    site = ingest.observatory_from_header(
        {k: v for k, v in frames[0].header_preview.items()})
    print(f"    site from header: {site}")

    # --------------------------------------------------------------- detection
    data, hdr, _, _, sat = ingest.read_pixels(man["files"][0], "G")
    cfg_p = phot_mod.PhotConfig(fwhm=4.0, thresh_sigma=5.0, gain=cfg.gain,
                                read_noise=cfg.read_noise, saturation=sat)
    srcs = phot_mod.detect_sources(data, cfg_p)
    print(f"[3] detected {len(srcs)} sources on frame 1")
    assert len(srcs) >= 6, "too few sources detected"

    fw = phot_mod.measure_fwhm(data, [(s["x"], s["y"]) for s in srcs[:8]])
    print(f"    measured FWHM = {fw:.2f} px (simulated {cfg.fwhm:.2f})")
    cfg_p.fwhm = fw or 4.0

    # Match detected sources against the simulator's frame-1 truth positions
    # (frame 1 already carries the un-guided pointing offset).
    def nearest(x, y):
        d = [np.hypot(s["x"] - x, s["y"] - y) for s in srcs]
        return int(np.argmin(d)), float(np.min(d))

    truth_stars = truth["stars_frame1"]
    print(f"    frame-1 pointing offset: dx={truth['frame1_offset'][0]:+.1f} "
          f"dy={truth['frame1_offset'][1]:+.1f} px")

    ti = tdist = None
    comps, comp_mags = [], {}
    for ts in truth_stars:
        si, d = nearest(ts["x"], ts["y"])
        if d > 3.0:
            continue
        if ts["target"]:
            ti, tdist = si, d
        elif ts["name"].startswith("comp") and si not in comps:
            comps.append(si)
            comp_mags[si] = ts["mag"]

    assert ti is not None, "target not matched to any detected source"
    assert len(comps) >= 3, f"only matched {len(comps)} comparison stars"
    print(f"    target = source {ti} ({tdist:.2f} px from truth); "
          f"comps = {comps} at V = {[round(comp_mags[c], 2) for c in comps]}")

    # -------------------------------------------------------------- photometry
    positions = [(s["x"], s["y"]) for s in srcs]
    res = phot_mod.run_photometry(frames, positions, cfg_p)
    print(f"[4] photometry on {len(res.jd)}/{len(frames)} frames "
          f"({len(res.failures)} failures)")
    assert len(res.jd) > 0.85 * len(frames), "too many frames dropped"
    drift = np.hypot(res.xpos[-1, ti] - res.xpos[0, ti],
                     res.ypos[-1, ti] - res.ypos[0, ti])
    print(f"    target tracked through {drift:.1f} px of drift")

    rep = phot_mod.comparison_report(res, ti, comps)
    for r in rep[:7]:
        if r["role"] != "unused":
            print(f"    star {r['star']:2d} {r['role']:11s} SNR={r['median_snr']:7.1f} "
                  f"rms={r['rms_mmag']:7.1f} mmag  check={r['check_rms_mmag']}")

    # Feed the comparison stars' catalog magnitudes in, so absolute calibration
    # is exercised too and the recovered mean V can be checked against truth.
    diff = phot_mod.differential(res, ti, comps, comp_mags=comp_mags, saturation=sat)
    good = diff["good"]
    print(f"[5] differential photometry: {good.sum()} good points, "
          f"scatter = {np.nanstd(diff['dmag'][good]) * 1000:.1f} mmag, "
          f"median sigma = {np.nanmedian(diff['sigma'][good]) * 1000:.1f} mmag")
    assert diff["calibrated"], "absolute calibration did not run"
    print(f"    zero point = {diff['zeropoint']:.4f} +- {diff['zeropoint_sigma']:.4f} "
          f"from {diff['n_calibrators']} calibrators")

    # ---------------------------------------------------------------- timing
    tt, label, delta = timing.convert_times(
        res.jd[good], timing.TARGET_DEFAULT["ra_deg"],
        timing.TARGET_DEFAULT["dec_deg"],
        {"lat": cfg.site["lat"], "lon": cfg.site["lon"], "elev": cfg.site["elev"]},
        "bjd_tdb")
    print(f"[6] time system: {label}; mean correction {delta:+.1f} s")

    y = diff["dmag"][good]
    dy = diff["sigma"][good]

    # ---------------------------------------------------------------- period
    ls = per_mod.lomb_scargle(tt, y, dy, p_min=0.02, p_max=0.30, oversample=30)
    print(f"[7] Lomb-Scargle: P = {ls['period_best']:.7f} d "
          f"({ls['period_best'] * 24:.5f} h), power = {ls['power_best']:.3f}, "
          f"FAP = {ls['fap']:.2e}")
    print(f"    Rayleigh resolution 1/T = {ls['rayleigh_df']:.4f} c/d")

    fit = per_mod.fourier_fit(tt, y, dy, ls["freq_best"], nharm=4)
    print(f"    Fourier: semi-amp = {fit['amp_semi']:.4f} mag, "
          f"p2p = {fit['amp_peak_to_peak']:.4f} mag, rms = {fit['rms'] * 1000:.1f} mmag, "
          f"chi2red = {fit['chi2_red']:.2f}")

    unc = per_mod.period_uncertainty(tt, fit["rms"], fit["amp_semi"], len(tt),
                                     ls["period_best"])
    print(f"    sigma_P (Montgomery & O'Donoghue) = {unc['sigma_period']:.2e} d "
          f"({unc['sigma_period'] / ls['period_best'] * 100:.3f}%)")

    boot = per_mod.bootstrap_period(tt, y, dy, ls["freq_best"], n_iter=120, nharm=3)
    print(f"    sigma_P (bootstrap, {boot['n_iter']} trials) = "
          f"{boot['sigma_period']:.2e} d")

    pdm_res = per_mod.pdm(tt, y, 0.02, 0.30, n_periods=2500, nbins=10, ncover=2)
    print(f"[8] PDM: P = {pdm_res['period_best']:.7f} d, "
          f"theta = {pdm_res['theta_best']:.4f}")

    span_cyc = (tt.max() - tt.min()) / ls["period_best"]
    cons = per_mod.consolidate_uncertainty(ls["period_best"], unc["sigma_period"],
                                           boot["sigma_period"],
                                           pdm_res["period_best"], span_cyc)
    print(f"    consolidated sigma_P = {cons['sigma_period']:.2e} d "
          f"({cons['sigma_period'] / ls['period_best'] * 100:.3f}%) "
          f"<- {cons['driver']}")

    modes = per_mod.prewhiten(tt, y, dy, 0.02, 0.30, n_modes=3, nharm=2)
    for m in modes:
        print(f"    mode {m['mode']}: P = {m['period']:.6f} d, "
              f"amp = {m['amp_mmag']:.1f} mmag, SNR = {m['snr']:.1f}, "
              f"significant = {m['significant']}")
    ratios = per_mod.classify_mode_ratio(modes)
    for r in ratios:
        print(f"    -> P ratio {r['ratio']:.4f}: fundamental = "
              f"{r['fundamental_period']:.6f} d")

    # ----- accuracy of the recovered period
    p_true = truth["period"]
    p_meas = ls["period_best"]
    err = abs(p_meas - p_true)
    print(f"\n[9] PERIOD ACCURACY: measured {p_meas:.7f} vs true {p_true:.7f} d")
    print(f"    absolute error {err:.2e} d = {err / p_true * 100:.4f}%")
    if np.isfinite(unc["sigma_period"]) and unc["sigma_period"] > 0:
        print(f"    that is {err / unc['sigma_period']:.2f} sigma")

    sigma_p = cons["sigma_period"]
    print(f"    true error is {err / sigma_p:.2f} consolidated sigma")
    assess = per_mod.assess(tt, p_meas, sigma_p, len(tt), ls["fap"],
                            fit["amp_semi"], fit["rms"])
    print(f"    verdict: {assess['verdict']}  ({assess['cycles']:.2f} cycles, "
          f"{assess['n_sessions']} session(s))")
    for f in assess["flags"][:4]:
        print(f"      [{f['level']:8s}] {f['text'][:110]}")

    # -------------------------------------------------------------- distance
    # Mean V from the Fourier fit of the *calibrated* light curve.
    cal = diff["mag"][good]
    fit_cal = per_mod.fourier_fit(tt, cal, dy, ls["freq_best"], nharm=4)
    mean_v = fit_cal["mag_mean_intensity"]
    sigma_v = float(np.hypot(diff["zeropoint_sigma"],
                             fit_cal["rms"] / np.sqrt(len(tt))))
    print(f"\n    calibrated <V> = {mean_v:.4f} +- {sigma_v:.4f} "
          f"(truth {truth['mean_mag']:.4f}, error "
          f"{mean_v - truth['mean_mag']:+.4f} mag)")
    print(f"    recovered p2p amplitude = {fit_cal['amp_peak_to_peak']:.4f} mag "
          f"(truth {truth['amplitude_p2p']:.4f})")
    assert abs(mean_v - truth["mean_mag"]) < 0.15, "mean magnitude badly off"

    sol = dist_mod.solve(p_meas, sigma_p, mean_v, sigma_v,
                         relation="ziaali2019", ebv=0.09)
    d = sol["distance"]
    print(f"\n[10] DISTANCE")
    print(f"    M_V = {sol['absolute']['M']:.3f} +- {sol['absolute']['sigma_M']:.3f}")
    print(f"    A_V = {sol['extinction']['a_v']:.3f}  ({sol['extinction']['source']})")
    print(f"    mu_0 = {d['mu_0']:.3f} +- {d['sigma_mu']:.3f}")
    print(f"    d = {d['distance_pc']:.0f} pc  "
          f"[{d['distance_pc_lo']:.0f}, {d['distance_pc_hi']:.0f}]  "
          f"= {d['distance_ly']:.0f} ly  ({d['error_budget']['relative_distance_pct']:.1f}%)")

    props = dist_mod.stellar_properties(sol["absolute"]["M"],
                                        sol["absolute"]["sigma_M"], p_meas, teff=7400)
    print(f"    L = {props['luminosity_lsun']:.1f} +- {props['sigma_luminosity_lsun']:.1f} Lsun, "
          f"R = {props['radius_rsun']:.2f} Rsun, M = {props['mass_msun']:.2f} Msun")

    # ---------------------------------------------------------------- plots
    from pipeline import plotting as pl
    phase = per_mod.phase_fold(tt, p_meas, fit["t_max"])
    outs = {
        "lightcurve": pl.light_curve(tt, y, dy, tt, fit["model"]),
        "periodogram": pl.periodogram(ls["freq"], ls["power"], ls["freq_best"],
                                      per_mod.top_peaks(ls["freq"], ls["power"],
                                                        span=ls["span"]),
                                      catalog_period=p_true,
                                      rayleigh=ls["rayleigh_df"]),
        "pdm": pl.pdm_plot(pdm_res["period"], pdm_res["theta"],
                           pdm_res["period_best"], p_true),
        "folded": pl.folded(phase, y, dy, per_mod.bin_phase(phase, y, 40),
                            fit["curve_phase"], fit["curve_mag"], p_meas),
        "diagnostics": pl.diagnostics(tt, res.fwhm[good], res.sky[good],
                                      res.xpos[good][:, ti], res.ypos[good][:, ti],
                                      diff["flux_comp"][good]),
        "distance": pl.distance_summary(sol),
        "plrelation": pl.pl_relation(sol),
        "bootstrap": pl.bootstrap_hist(boot["periods"], p_meas),
    }
    plotdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selftest_plots")
    os.makedirs(plotdir, exist_ok=True)
    for name, png in outs.items():
        p = os.path.join(plotdir, f"{name}.png")
        open(p, "wb").write(png)
    print(f"\n[11] wrote {len(outs)} plots to {plotdir}")

    shutil.rmtree(workdir, ignore_errors=True)
    print(f"\n# completed in {time.time() - t_start:.1f}s")

    ok = err / p_true < 0.02
    print(f"# RESULT: {'PASS' if ok else 'FAIL'} (period recovered to "
          f"{err / p_true * 100:.3f}%)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "clean"))
