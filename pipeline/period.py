"""Period determination: Lomb-Scargle, phase dispersion minimisation, Fourier fits.

Two independent period estimators are run on every data set because they fail in
different ways: Lomb-Scargle fits a sinusoid (excellent signal-to-noise, biased
when the light curve is strongly non-sinusoidal, as HADS stars are), while PDM
makes no assumption about curve shape (robust, coarser). Agreement between them
is the main internal consistency check.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from astropy.timeseries import LombScargle

# Radial double-mode pulsators (HADS(B), the class of V0756 CrA) show a
# first-overtone / fundamental period ratio in a narrow band.
P1_P0_RATIO = (0.756, 0.787)


def _clean(t, y, dy=None):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    if dy is not None:
        dy = np.asarray(dy, float)
        m &= np.isfinite(dy) & (dy > 0)
        return t[m], y[m], dy[m], m
    return t[m], y[m], None, m


def detrend(t, y, order: int = 0):
    """Remove a low-order polynomial in time (slow transparency/airmass drifts).

    Order 0 subtracts only the mean. Beware: a polynomial of order >= 2 over a
    baseline comparable to the period will eat real pulsation signal.
    """
    if order is None or order < 0:
        return y, None
    if len(t) <= order + 1:
        return y - np.mean(y), None
    tt = t - np.mean(t)
    coef = np.polyfit(tt, y, order)
    return y - np.polyval(coef, tt), coef


# --------------------------------------------------------------------------
# Lomb-Scargle
# --------------------------------------------------------------------------

def lomb_scargle(t, y, dy=None, p_min=0.01, p_max=1.0, oversample=25,
                 nterms=1, normalization="standard"):
    """Periodogram over a period range, plus the refined peak.

    Returns a dict with the frequency grid, power, best period and a parabolic
    sub-grid refinement of the peak.
    """
    t, y, dy, _ = _clean(t, y, dy)
    n = len(t)
    if n < 6:
        raise ValueError(f"need at least 6 photometric points, have {n}")

    span = float(t.max() - t.min())
    if span <= 0:
        raise ValueError("all measurements share the same timestamp")

    p_min = max(float(p_min), 1e-4)
    p_max = min(float(p_max), 10.0 * span if span > 0 else float(p_max))
    if p_max <= p_min:
        p_max = p_min * 10

    f_min, f_max = 1.0 / p_max, 1.0 / p_min
    df = 1.0 / (oversample * span)
    nf = int(np.ceil((f_max - f_min) / df)) + 1
    nf = int(min(max(nf, 256), 4_000_000))
    freq = np.linspace(f_min, f_max, nf)

    ls = LombScargle(t, y, dy, nterms=nterms, normalization=normalization)
    power = ls.power(freq)

    k = int(np.nanargmax(power))
    f_peak = float(freq[k])

    # Parabolic interpolation through the three grid points at the peak.
    if 0 < k < len(freq) - 1:
        y0, y1, y2 = power[k - 1], power[k], power[k + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom != 0:
            shift = 0.5 * (y0 - y2) / denom
            if abs(shift) <= 1.0:
                f_peak = float(freq[k] + shift * (freq[1] - freq[0]))

    # Then a dense local grid, which beats parabolic interpolation for
    # sharp peaks from long baselines.
    half = 5.0 / span
    fine = np.linspace(max(f_min, f_peak - half), min(f_max, f_peak + half), 20001)
    fine_power = ls.power(fine)
    kf = int(np.nanargmax(fine_power))
    f_best = float(fine[kf])
    p_best = 1.0 / f_best

    # The analytic false-alarm probability needs an explicit frequency range;
    # left to guess, astropy's auto-range can disagree with the grid actually
    # searched and refuse. Pass the same limits used above.
    peak_power = float(np.nanmax(power))
    fap, fap_error = float("nan"), None
    for method in ("baluev", "naive"):
        try:
            val = float(ls.false_alarm_probability(
                peak_power, method=method,
                minimum_frequency=f_min, maximum_frequency=f_max))
            # Baluev's expression returns NaN rather than raising when the power
            # sits close enough to 1 that the analytic terms underflow.
            if np.isfinite(val):
                fap, fap_error = val, None
                break
            fap_error = f"{method}: returned NaN (peak power {peak_power:.6f})"
        except Exception as exc:
            fap_error = f"{method}: {type(exc).__name__}: {exc}"

    if not np.isfinite(fap) and peak_power > 0.5:
        # Underflow only happens for overwhelmingly significant peaks; saying so
        # is more useful than reporting nothing.
        fap = 0.0
        fap_error = (f"analytic formulae underflowed at peak power "
                     f"{peak_power:.4f}; the false-alarm probability is below "
                     f"double-precision resolution (< 1e-300)")

    return {
        "freq": freq, "power": power,
        "freq_best": f_best, "period_best": p_best,
        "power_best": float(np.nanmax(fine_power)),
        "fap": fap, "fap_error": fap_error,
        "span": span, "n": n,
        "rayleigh_df": 1.0 / span,
        "ls": ls,
        "nterms": nterms,
        "normalization": normalization,
    }


def top_peaks(freq, power, n_peaks=5, min_sep_df=None, span=None):
    """Distinct local maxima, strongest first - for spotting aliases."""
    power = np.asarray(power, float)
    if min_sep_df is None:
        min_sep_df = (1.0 / span) if span else (freq[1] - freq[0]) * 10
    order = np.argsort(power)[::-1]
    picks = []
    for i in order:
        f = freq[i]
        if any(abs(f - pf) < min_sep_df for pf, _ in picks):
            continue
        picks.append((float(f), float(power[i])))
        if len(picks) >= n_peaks:
            break
    return [{"freq": f, "period": 1.0 / f, "power": p} for f, p in picks]


# --------------------------------------------------------------------------
# Fourier model at a fixed frequency
# --------------------------------------------------------------------------

def fourier_fit(t, y, dy=None, freq=None, nharm=3):
    """Weighted least-squares harmonic fit y = m0 + sum A_k sin(2pi k f t + ph_k).

    Gives the mean magnitude and amplitude that the period-luminosity relation
    needs, plus the epoch of maximum light.
    """
    t, y, dy, _ = _clean(t, y, dy)
    n = len(t)
    nharm = int(max(1, nharm))
    while nharm > 1 and n < 2 * nharm + 4:
        nharm -= 1

    t0 = float(np.min(t))
    tt = t - t0
    cols = [np.ones_like(tt)]
    for k in range(1, nharm + 1):
        arg = 2 * np.pi * k * freq * tt
        cols += [np.sin(arg), np.cos(arg)]
    A = np.vstack(cols).T

    if dy is not None and np.all(dy > 0):
        w = 1.0 / dy
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    else:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)

    model = A @ coef
    resid = y - model
    dof = max(1, n - len(coef))
    rms = float(np.sqrt(np.sum(resid ** 2) / dof))
    if dy is not None and np.all(dy > 0):
        chi2 = float(np.sum((resid / dy) ** 2))
        chi2_red = chi2 / dof
    else:
        chi2 = chi2_red = float("nan")

    amps, phases = [], []
    for k in range(nharm):
        a, b = coef[1 + 2 * k], coef[2 + 2 * k]
        amps.append(float(np.hypot(a, b)))
        phases.append(float(np.arctan2(b, a)))

    # Model curve over one cycle, to get true extrema and the mean.
    ph = np.linspace(0, 1, 2001, endpoint=False)
    cols_m = [np.ones_like(ph)]
    for k in range(1, nharm + 1):
        arg = 2 * np.pi * k * ph
        cols_m += [np.sin(arg), np.cos(arg)]
    curve = np.vstack(cols_m).T @ coef

    i_max = int(np.argmin(curve))   # brightest = smallest magnitude
    i_min = int(np.argmax(curve))
    mag_mean = float(np.mean(curve))
    # Intensity-weighted mean magnitude, the convention in P-L calibrations.
    with np.errstate(over="ignore"):
        inten_mean = float(-2.5 * np.log10(np.mean(10 ** (-0.4 * curve))))

    # Epoch of maximum light nearest the middle of the run.
    period = 1.0 / freq
    phase_max = float(ph[i_max])
    t_mid = float(np.median(t))
    cyc = np.round((t_mid - (t0 + phase_max * period)) / period)
    t_max = t0 + phase_max * period + cyc * period

    # Re-express the model curve with phase 0 at maximum light, which is the
    # origin `phase_fold(t, period, epoch=t_max)` uses for the data. Without
    # this shift the curve is drawn displaced from the points it fits, because
    # the design matrix measures phase from t0 = min(t) instead.
    ph_from_max = (ph - phase_max) % 1.0
    order_ph = np.argsort(ph_from_max)
    curve_phase_out = ph_from_max[order_ph]
    curve_mag_out = curve[order_ph]

    return {
        "coef": coef.tolist(), "nharm": nharm, "t0": t0, "freq": float(freq),
        "period": float(period),
        "mag_mean_fourier": float(coef[0]),
        "mag_mean_curve": mag_mean,
        "mag_mean_intensity": inten_mean,
        "mag_mean_data": float(np.mean(y)),
        "amp_semi": amps[0] if amps else float("nan"),
        "amp_peak_to_peak": float(curve[i_min] - curve[i_max]),
        "mag_at_max": float(curve[i_max]),
        "mag_at_min": float(curve[i_min]),
        "harmonic_amps": amps, "harmonic_phases": phases,
        "rms": rms, "chi2": chi2, "chi2_red": chi2_red,
        "n": n, "resid": resid, "model": model,
        "t_max": float(t_max), "phase_max": phase_max,
        # Phase 0 = maximum light, matching phase_fold(..., epoch=t_max).
        "curve_phase": curve_phase_out.tolist(),
        "curve_mag": curve_mag_out.tolist(),
    }


def eval_fourier(fit, t):
    coef = np.asarray(fit["coef"], float)
    nharm, t0, freq = fit["nharm"], fit["t0"], fit["freq"]
    tt = np.asarray(t, float) - t0
    cols = [np.ones_like(tt)]
    for k in range(1, nharm + 1):
        arg = 2 * np.pi * k * freq * tt
        cols += [np.sin(arg), np.cos(arg)]
    return np.vstack(cols).T @ coef


# --------------------------------------------------------------------------
# uncertainties
# --------------------------------------------------------------------------

def period_uncertainty(t, resid_rms, amplitude, n, period):
    """Analytic frequency error from Montgomery & O'Donoghue (1999).

    sigma_f = sqrt(6/N) * sigma_resid / (pi * T * A)   ->   sigma_P = P^2 * sigma_f

    This is a *lower bound*: it assumes white noise and a single coherent mode,
    and it says nothing about which alias peak you picked.
    """
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    span = float(t.max() - t.min()) if len(t) > 1 else 0.0
    if span <= 0 or amplitude <= 0 or n < 4:
        return {"sigma_freq": float("nan"), "sigma_period": float("nan"),
                "span": span, "rayleigh_period": float("nan")}
    sigma_f = np.sqrt(6.0 / n) * resid_rms / (np.pi * span * amplitude)
    return {
        "sigma_freq": float(sigma_f),
        "sigma_period": float(period ** 2 * sigma_f),
        "span": span,
        # Frequency resolution of the data set: the width of one peak.
        "rayleigh_period": float(period ** 2 / span),
    }


def detect_aliases(freq, power, freq_best, span, catalog_period=None,
                   power_tol=0.12, spacings=(1.0, 0.5), n_search=14):
    """Find competing periodogram peaks that the data cannot rule out.

    Observing from one site on a 24 h planet imprints a 1 cycle/day pattern on the
    sampling, so every real frequency f is shadowed by peaks at f +- n c/d. When a
    shadow peak carries almost as much power as the winner, the period is
    *ambiguous* rather than merely uncertain - and the honest output is a short
    list of candidates, not one number with a small error bar.

    The trap this catches: adding more short nights raises every alias together
    and never resolves the ambiguity. Only covering a full pulsation cycle inside
    one continuous session does, because then the shape of the curve itself
    constrains the frequency.
    """
    peaks = top_peaks(freq, power, n_peaks=n_search, span=span)
    if not peaks:
        return {"ambiguous": False, "candidates": [], "note": None}

    best_pow = peaks[0]["power"]
    if best_pow <= 0:
        return {"ambiguous": False, "candidates": [], "note": None}

    candidates = [{
        "freq": float(freq_best), "period": float(1.0 / freq_best),
        "power": float(best_pow), "rel_deficit": 0.0,
        "relation": "highest peak", "is_best": True,
    }]

    for p in peaks[1:]:
        rel = (best_pow - p["power"]) / best_pow
        if rel > power_tol:
            continue
        df = abs(p["freq"] - freq_best)
        rel_label = None
        for s in spacings:
            k = df / s
            if abs(k - round(k)) < 0.08 and round(k) >= 1:
                n = int(round(k))
                sign = "+" if p["freq"] > freq_best else "-"
                rel_label = (f"{sign}{n} x {s:g} cycle/day alias"
                             if s == 1.0 else
                             f"{sign}{n} x {s:g} c/d alias")
                break
        if rel_label is None:
            # A close, unexplained competitor still matters.
            if df < 3.0 / span:
                rel_label = "unresolved neighbouring peak"
            else:
                rel_label = "independent competing peak"
        candidates.append({
            "freq": float(p["freq"]), "period": float(p["period"]),
            "power": float(p["power"]), "rel_deficit": float(rel),
            "relation": rel_label, "is_best": False,
        })

    # Flag any candidate that agrees with a published period.
    if catalog_period:
        for c in candidates:
            frac = abs(c["period"] - catalog_period) / catalog_period
            c["matches_catalog"] = bool(frac < 0.01)
            c["catalog_diff_pct"] = float(frac * 100)

    ambiguous = len(candidates) > 1
    note = None
    if ambiguous:
        second = candidates[1]
        note = (
            f"The strongest competitor is a {second['relation']} at "
            f"P = {second['period']:.6f} d carrying "
            f"{100 * (1 - second['rel_deficit']):.1f}% of the peak power. "
            "The data cannot cleanly separate these.")
        match = next((c for c in candidates if c.get("matches_catalog")), None)
        if match and not match["is_best"]:
            note += (f" Note that the candidate at P = {match['period']:.6f} d is the "
                     "one matching the published period - the periodogram is very "
                     "likely favouring an alias here.")
    return {
        "ambiguous": ambiguous,
        "candidates": candidates,
        "n_candidates": len(candidates),
        "note": note,
        "power_tol": power_tol,
    }


def consolidate_uncertainty(period_ls, sigma_formal, sigma_boot=None,
                            period_pdm=None, cycles=None):
    """Choose a defensible period uncertainty from several independent estimates.

    The analytic formula assumes white noise, one coherent sinusoid, and that the
    right periodogram peak was picked. When a run covers only a cycle or two none
    of those hold: the peak position is pulled around by the observing window, by
    the non-sinusoidal curve shape and by any second mode, and the formal error
    can understate the real one by more than an order of magnitude.

    So the largest of the available estimates is quoted, including the outright
    disagreement between Lomb-Scargle and PDM - two estimators with different
    assumptions, whose spread is an empirical handle on systematic error.
    """
    estimates = []

    # Below about 1.5 cycles the period is degenerate with the length of the run:
    # the periodogram peak is barely one peak-width wide, nothing verifies that
    # the curve repeats, and a wrong peak can be picked with no symptom in the
    # noise. In that regime the frequency resolution of the data set, dP = P^2/T,
    # is the honest statement of what is known - noise-based formulae are simply
    # answering the wrong question and can be optimistic by a factor of 50.
    if cycles is not None and cycles < 1.5 and period_ls > 0:
        span = period_ls * cycles
        if span > 0:
            estimates.append({
                "method": "Frequency-resolution limit (1/T)",
                "sigma": float(period_ls ** 2 / span),
                "note": ("Applied because the run covers fewer than 1.5 cycles, so "
                         "the period is not separable from the length of the run."),
            })

    if sigma_formal is not None and np.isfinite(sigma_formal) and sigma_formal > 0:
        estimates.append({
            "method": "Analytic (Montgomery & O'Donoghue 1999)",
            "sigma": float(sigma_formal),
            "note": "Noise-only. A lower bound, not an error bar.",
        })
    if sigma_boot is not None and np.isfinite(sigma_boot) and sigma_boot > 0:
        estimates.append({
            "method": "Residual bootstrap",
            "sigma": float(sigma_boot),
            "note": "Randomises the noise but keeps the observing window.",
        })

    disagreement = None
    if period_pdm and np.isfinite(period_pdm) and period_ls > 0:
        frac = abs(period_pdm - period_ls) / period_ls
        if frac < 0.25:      # same peak, not a different alias
            disagreement = abs(period_pdm - period_ls)
            estimates.append({
                "method": "Lomb-Scargle vs PDM disagreement",
                "sigma": float(disagreement),
                "note": ("Spread between two estimators with different "
                         "assumptions - captures curve-shape and window bias "
                         "that the noise-only formulae miss."),
            })

    if not estimates:
        return {"sigma_period": float("nan"), "estimates": [], "driver": None,
                "rationale": "No uncertainty estimate could be formed."}

    best = max(estimates, key=lambda e: e["sigma"])
    sigma = best["sigma"]

    if cycles is not None and cycles < 3 and len(estimates) > 1:
        rationale = (
            f"Only {cycles:.2f} cycles were observed, so the analytic error is "
            "not trustworthy on its own. The largest of the independent "
            f"estimates is quoted: {best['method']}.")
    else:
        rationale = f"Largest of the independent estimates: {best['method']}."

    return {
        "sigma_period": float(sigma),
        "estimates": estimates,
        "driver": best["method"],
        "pdm_disagreement": float(disagreement) if disagreement is not None else None,
        "rationale": rationale,
    }


def bootstrap_period(t, y, dy, freq_best, n_iter=200, window_df=None,
                     nharm=3, seed=12345):
    """Residual-bootstrap period error.

    Residuals are resampled around the best-fit model and the peak is re-found,
    which preserves the observing window (and therefore the alias structure)
    while randomising the noise.
    """
    t, y, dy, _ = _clean(t, y, dy)
    n = len(t)
    if n < 8:
        return {"sigma_period": float("nan"), "periods": [], "n_iter": 0}
    span = float(t.max() - t.min())
    if window_df is None:
        window_df = 3.0 / span

    fit = fourier_fit(t, y, dy, freq_best, nharm=nharm)
    model = np.asarray(fit["model"], float)
    resid = np.asarray(fit["resid"], float)

    rng = np.random.default_rng(seed)
    f_lo = max(1e-6, freq_best - window_df)
    f_hi = freq_best + window_df
    grid = np.linspace(f_lo, f_hi, 4001)

    periods = []
    for _ in range(int(n_iter)):
        yb = model + rng.choice(resid, size=n, replace=True)
        try:
            p = LombScargle(t, yb, dy).power(grid)
        except Exception:
            continue
        periods.append(1.0 / float(grid[int(np.nanargmax(p))]))

    if len(periods) < 10:
        return {"sigma_period": float("nan"), "periods": periods, "n_iter": len(periods)}
    arr = np.asarray(periods)
    return {
        "sigma_period": float(np.std(arr, ddof=1)),
        "period_median": float(np.median(arr)),
        "p16": float(np.percentile(arr, 15.865)),
        "p84": float(np.percentile(arr, 84.135)),
        "periods": arr.tolist(),
        "n_iter": len(periods),
    }


# --------------------------------------------------------------------------
# phase dispersion minimisation
# --------------------------------------------------------------------------

def pdm(t, y, p_min, p_max, n_periods=6000, nbins=10, ncover=2):
    """Stellingwerf (1978) theta statistic. Minima mark candidate periods.

    theta = (pooled within-bin variance) / (overall variance); theta -> 0 for a
    period that phases the data into a tight curve, theta ~ 1 for noise.
    """
    t, y, _, _ = _clean(t, y)
    if len(t) < 10:
        return {"period": np.array([]), "theta": np.array([]),
                "period_best": float("nan"), "theta_best": float("nan")}

    span = float(t.max() - t.min())
    p_max = min(float(p_max), span if span > 0 else float(p_max))
    p_min = max(float(p_min), 1e-4)
    if p_max <= p_min:
        p_max = p_min * 5

    # Uniform in frequency so resolution is even across the range.
    freqs = np.linspace(1.0 / p_max, 1.0 / p_min, int(n_periods))
    periods = 1.0 / freqs

    var_all = float(np.var(y, ddof=1))
    if var_all <= 0:
        return {"period": periods, "theta": np.ones_like(periods),
                "period_best": float("nan"), "theta_best": float("nan")}

    n_total = len(y)
    offsets = np.arange(ncover) / (nbins * ncover)
    thetas = np.empty_like(periods)

    for i, p in enumerate(periods):
        phase = (t / p) % 1.0
        num = 0.0
        den = 0
        for off in offsets:
            ph = (phase + off) % 1.0
            idx = np.minimum((ph * nbins).astype(int), nbins - 1)
            for b in range(nbins):
                sel = y[idx == b]
                nb = sel.size
                if nb > 1:
                    num += (nb - 1) * np.var(sel, ddof=1)
                    den += nb - 1
        thetas[i] = (num / den) / var_all if den > 0 else 1.0

    k = int(np.nanargmin(thetas))
    # Refine with a dense local scan.
    p_best = float(periods[k])
    lo, hi = p_best * 0.995, p_best * 1.005
    fine = np.linspace(lo, hi, 601)
    fine_theta = []
    for p in fine:
        phase = (t / p) % 1.0
        num, den = 0.0, 0
        for off in offsets:
            ph = (phase + off) % 1.0
            idx = np.minimum((ph * nbins).astype(int), nbins - 1)
            for b in range(nbins):
                sel = y[idx == b]
                if sel.size > 1:
                    num += (sel.size - 1) * np.var(sel, ddof=1)
                    den += sel.size - 1
        fine_theta.append((num / den) / var_all if den > 0 else 1.0)
    fine_theta = np.asarray(fine_theta)
    kf = int(np.nanargmin(fine_theta))

    return {
        "period": periods, "theta": thetas,
        "period_best": float(fine[kf]), "theta_best": float(fine_theta[kf]),
        "nbins": nbins, "ncover": ncover,
    }


# --------------------------------------------------------------------------
# multi-mode prewhitening
# --------------------------------------------------------------------------

def prewhiten(t, y, dy=None, p_min=0.01, p_max=1.0, n_modes=3, nharm=2,
              oversample=25, snr_floor=3.5):
    """Iteratively extract and subtract the strongest modes.

    V0756 CrA is classified HADS(B) - a *double-mode* high-amplitude delta Scuti -
    so a second independent frequency is expected, and identifying which mode is
    the radial fundamental decides which period feeds the P-L relation.
    """
    t, y, dy, _ = _clean(t, y, dy)
    resid = y.copy()
    modes = []
    for i in range(int(n_modes)):
        try:
            res = lomb_scargle(t, resid, dy, p_min=p_min, p_max=p_max,
                               oversample=oversample)
        except ValueError:
            break
        fit = fourier_fit(t, resid, dy, res["freq_best"], nharm=nharm)
        noise = float(np.std(fit["resid"])) if len(fit["resid"]) > 2 else np.nan
        amp = fit["amp_semi"]
        snr = float(amp / noise) if noise and noise > 0 else float("nan")
        modes.append({
            "mode": i + 1,
            "freq": res["freq_best"],
            "period": res["period_best"],
            "amp_semi": amp,
            "amp_mmag": amp * 1000.0,
            "power": res["power_best"],
            "fap": res["fap"],
            "snr": snr,
            "significant": bool(np.isfinite(snr) and snr >= snr_floor),
        })
        resid = np.asarray(fit["resid"], float)
        if np.isfinite(snr) and snr < snr_floor:
            break
    return modes


def classify_mode_ratio(modes, snr_min=3.0):
    """Check pairs of periods against the radial double-mode ratio P1/P0.

    Modes down to S/N ~ 3 are considered, not just formally significant ones: a
    secondary mode landing inside the narrow 0.756-0.787 window is itself strong
    evidence, since noise has no reason to prefer that ratio. Pairs where either
    mode is marginal are labelled tentative rather than hidden.
    """
    out = []
    cand = [m for m in modes
            if m.get("snr") is not None and np.isfinite(m["snr"]) and m["snr"] >= snr_min]
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a, b = cand[i], cand[j]
            p_long, p_short = max(a["period"], b["period"]), min(a["period"], b["period"])
            ratio = p_short / p_long
            if not (P1_P0_RATIO[0] <= ratio <= P1_P0_RATIO[1]):
                continue
            firm = bool(a.get("significant") and b.get("significant"))
            text = ("Period ratio matches the radial fundamental / first-overtone band "
                    "(0.756-0.787) expected for a double-mode HADS - which is exactly "
                    "what the (B) in this star's HADS(B) classification means. The "
                    "longer period is the fundamental, and that is the one the "
                    "period-luminosity relation is calibrated on.")
            if not firm:
                text += (f" Note that the weaker mode is only detected at S/N "
                         f"{min(a['snr'], b['snr']):.1f}, so treat this as tentative - "
                         "confirming it needs more data, not more analysis.")
            out.append({
                "modes": [a["mode"], b["mode"]],
                "ratio": float(ratio),
                "fundamental_period": float(p_long),
                "overtone_period": float(p_short),
                "confidence": "firm" if firm else "tentative",
                "min_snr": float(min(a["snr"], b["snr"])),
                "interpretation": text,
            })
    return out


# --------------------------------------------------------------------------
# quality assessment
# --------------------------------------------------------------------------

def session_stats(t, gap_threshold=0.25):
    """Split the time series into observing sessions separated by real gaps."""
    t = np.sort(np.asarray(t, float))
    t = t[np.isfinite(t)]
    if len(t) < 2:
        return [], np.zeros(len(t), dtype=int)
    breaks = np.where(np.diff(t) > gap_threshold)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(t) - 1]])
    labels = np.zeros(len(t), dtype=int)
    sessions = []
    for k, (a, b) in enumerate(zip(starts, ends)):
        labels[a:b + 1] = k
        sessions.append({"index": k, "n": int(b - a + 1),
                         "start": float(t[a]), "end": float(t[b]),
                         "span_days": float(t[b] - t[a])})
    return sessions, labels


def assess(t, period, sigma_period, n, fap, amp, resid_rms, comps_ok=True,
           aliases=None):
    """Plain-language verdict on how far the data can actually be trusted."""
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    span = float(t.max() - t.min()) if len(t) > 1 else 0.0
    cycles = span / period if period > 0 else 0.0
    gaps = np.diff(np.sort(t))
    cadence = float(np.median(gaps)) if len(gaps) else float("nan")
    sessions, _ = session_stats(t)
    n_nights = max(1, len(sessions))
    longest = max((s["span_days"] for s in sessions), default=span)
    cycles_longest = longest / period if period > 0 else 0.0

    flags = []
    if cycles < 1.5:
        flags.append({
            "level": "critical",
            "text": (f"Your data span {span * 24:.2f} h = {cycles:.2f} pulsation cycles. "
                     "Below about 1.5 cycles the period is not measured, only guessed "
                     "at: the periodogram peak is set by the length of the run rather "
                     "than by the star, and nothing in the data confirms that the "
                     "brightness pattern repeats. Note that this cycle count is "
                     "computed from the period found - if that period is itself wrong, "
                     "the coverage looks better than it is, which is exactly why the "
                     "quoted error bar cannot be trusted here."),
        })
    elif cycles < 3.0:
        flags.append({
            "level": "warning",
            "text": (f"Only {cycles:.1f} cycles covered ({span * 24:.2f} h). A single peak "
                     "can be fitted but the period error is dominated by the short "
                     "baseline. Two or three more nights would shrink it by roughly the "
                     "ratio of the baselines."),
        })
    else:
        flags.append({
            "level": "ok",
            "text": f"{cycles:.1f} cycles covered over {span * 24:.2f} h - enough to fit a period.",
        })

    # The single most useful diagnostic: cycles covered within one continuous
    # session. This, not the total number of nights, is what breaks the 1 c/d
    # alias - because only the shape of the curve inside an uninterrupted run
    # can distinguish f from f +- 1 c/d.
    if cycles_longest < 1.0:
        flags.append({
            "level": "critical" if n_nights > 1 else "warning",
            "text": (f"Your longest continuous session covers only "
                     f"{cycles_longest:.2f} of a cycle ({longest * 24:.2f} h vs a "
                     f"{period * 24:.2f} h period). This is what creates the "
                     "1 cycle/day ambiguity, and adding more short nights will NOT "
                     "fix it - every alias grows together. Extend a single run past "
                     f"{period * 24:.2f} h of continuous imaging instead."),
        })
    elif cycles_longest < 1.5:
        flags.append({
            "level": "warning",
            "text": (f"Longest session covers {cycles_longest:.2f} cycles - just "
                     "enough to start breaking the 1 cycle/day alias. Aim for at "
                     f"least {1.5 * period * 24:.1f} h continuous to be safe."),
        })
    else:
        flags.append({
            "level": "ok",
            "text": (f"Longest continuous session covers {cycles_longest:.2f} cycles, "
                     "which constrains the frequency from the curve shape and "
                     "suppresses the 1 cycle/day alias."),
        })

    if n_nights == 1 and cycles >= 2:
        flags.append({
            "level": "info",
            "text": ("Single-session data, so there is no 1 cycle/day alias structure "
                     "from night-to-night gaps - but also no long baseline, so the "
                     "period precision is limited by the length of this one run."),
        })
    if n_nights > 1:
        span_txt = (f"{n_nights} observing sessions over {span:.2f} d. The long "
                    "baseline sharpens the peak, at the cost of 1 cycle/day aliases.")
        flags.append({"level": "info", "text": span_txt})

    alias_deficit = None
    if aliases and aliases.get("ambiguous"):
        cands = aliases["candidates"]
        alias_deficit = cands[1]["rel_deficit"]
        listing = ", ".join(f"{c['period']:.6f} d" for c in cands[:4])
        matched = next((c for c in cands if c.get("matches_catalog")), None)
        # How decisively the winner beats its closest rival decides how much this
        # matters: a 0.5% margin is a coin flip, a 6% margin is a clear preference.
        if alias_deficit < 0.01:
            lvl = "critical"
            head = ("PERIOD AMBIGUOUS - the top two peaks are within 1% of each "
                    "other, which is a coin flip")
        elif alias_deficit < 0.05:
            lvl = "warning"
            head = ("Period favoured but not established - a competing alias holds "
                    f"{100 * (1 - alias_deficit):.1f}% of the peak power")
        else:
            lvl = "info"
            head = ("Aliases are present but suppressed; the chosen peak leads by "
                    f"{alias_deficit * 100:.1f}%")
        flags.append({
            "level": lvl,
            "text": (f"{head}. Candidates: {listing}. " + (aliases.get("note") or "")),
        })
        if matched:
            flags.append({
                "level": "info",
                "text": (f"One candidate ({matched['period']:.6f} d) agrees with the "
                         f"published period to {matched['catalog_diff_pct']:.2f}%. "
                         "That is independent evidence for which alias is real, but "
                         "quoting it means you are relying on the catalog, not on "
                         "your own data - say so explicitly if you do."),
            })

    if np.isfinite(fap):
        if fap < 1e-4:
            flags.append({"level": "ok",
                          "text": f"Peak false-alarm probability {fap:.2e} - the signal is real."})
        elif fap < 0.01:
            flags.append({"level": "warning",
                          "text": f"False-alarm probability {fap:.3f} - marginal detection."})
        else:
            flags.append({"level": "critical",
                          "text": (f"False-alarm probability {fap:.2f}. This periodogram peak is "
                                   "consistent with noise; do not report a period from it.")})

    snr = amp / resid_rms if resid_rms and resid_rms > 0 else float("nan")
    if np.isfinite(snr):
        if snr < 3:
            flags.append({"level": "critical",
                          "text": f"Amplitude / scatter = {snr:.1f}. The variation is buried in noise."})
        elif snr < 8:
            flags.append({"level": "warning",
                          "text": f"Amplitude / scatter = {snr:.1f}. Usable but noisy."})
        else:
            flags.append({"level": "ok",
                          "text": f"Amplitude / scatter = {snr:.0f} - strong, clean variation."})

    if n < 30:
        flags.append({"level": "warning",
                      "text": f"Only {n} usable frames. The proposal's continuous 2-3 h run at "
                              "30-45 s cadence should yield 150-350."})

    if not comps_ok:
        flags.append({"level": "warning",
                      "text": "At least one comparison star is itself variable or noisy - see the "
                              "check-star scatter column."})

    rel = (sigma_period / period * 100) if (period and np.isfinite(sigma_period)) else float("nan")
    if np.isfinite(rel):
        if rel > 20:
            lvl = "critical"
        elif rel > 5:
            lvl = "warning"
        else:
            lvl = "ok"
        # d(log P) -> d(M_V) -> fractional distance:
        # sigma_d/d = (ln10/5) * |slope| * sigma_P/(P ln10) = |slope|/5 * (sigma_P/P)
        dist_pct = abs(-2.94) / 5.0 * rel
        flags.append({"level": lvl,
                      "text": (f"Quoted period precision {rel:.2f}%. Through the "
                               f"period-luminosity relation this alone contributes "
                               f"about {dist_pct:.2f}% to the distance error.")})

    if cycles < 3 and np.isfinite(rel):
        flags.append({
            "level": "warning",
            "text": ("With this few cycles the quoted period error is dominated by "
                     "systematics, not noise. Treat it as indicative: the analytic "
                     "formula assumes a single sinusoid observed over many cycles, "
                     "and a short run violates both. The honest test is whether an "
                     "independent night phases up on the same period."),
        })

    order = {"critical": 0, "warning": 1, "info": 2, "ok": 3}
    flags.sort(key=lambda f: order.get(f["level"], 9))

    # The verdict is scored against the conditions that actually decide whether a
    # period can be quoted, rather than just echoing the most alarming flag.
    fatal = (
        # Fewer than ~1.5 cycles means the period cannot be checked for
        # repetition and is degenerate with the run length. Measured on this
        # star's data, single runs in that regime came out wrong by 3-52%,
        # while the quoted noise-based error stayed under 1%.
        cycles < 1.5
        or (np.isfinite(fap) and fap > 0.01)
        or (np.isfinite(snr) and snr < 3)
        or (alias_deficit is not None and alias_deficit < 0.01)
        or not np.isfinite(period)
    )
    shaky = (
        cycles < 3.0
        or (alias_deficit is not None and alias_deficit < 0.05)
        or (np.isfinite(rel) and rel > 10)
        # A second pulsation mode left in the residuals inflates the scatter, so
        # this threshold is deliberately loose.
        or (np.isfinite(snr) and snr < 6)
        or cycles_longest < 1.0
        or n < 30
    )
    marginal = (
        cycles < 10.0
        or (np.isfinite(rel) and rel > 3)
        or not comps_ok
        or cycles_longest < 1.5
    )
    if fatal:
        verdict = "Insufficient"
        verdict_note = ("The data do not support quoting a period. Fix the cause "
                        "flagged in red before going on to a distance.")
    elif shaky:
        verdict = "Provisional"
        verdict_note = ("A period can be quoted with its caveats, but do not treat "
                        "the error bar as final.")
    elif marginal:
        verdict = "Reasonable"
        verdict_note = "Sound enough for a course report, with the stated uncertainty."
    else:
        verdict = "Good"
        verdict_note = "Well-constrained period; the distance error will be dominated "\
                       "by the period-luminosity relation, not by your data."

    return {
        "span_days": span, "span_hours": span * 24.0, "cycles": cycles,
        "cadence_s": cadence * 86400.0 if np.isfinite(cadence) else None,
        "n_points": int(n), "n_sessions": n_nights,
        "longest_session_hours": longest * 24.0,
        "cycles_in_longest_session": cycles_longest,
        "sessions": sessions,
        "amp_over_rms": float(snr) if np.isfinite(snr) else None,
        "rel_period_error_pct": float(rel) if np.isfinite(rel) else None,
        "flags": flags, "verdict": verdict, "verdict_note": verdict_note,
        "alias_power_margin_pct": (alias_deficit * 100) if alias_deficit is not None else None,
    }


def phase_fold(t, period, epoch=None):
    t = np.asarray(t, float)
    if epoch is None:
        epoch = float(np.nanmin(t))
    return ((t - epoch) / period) % 1.0


def bin_phase(phase, mag, nbins=40):
    """Phase-binned means with standard errors, for the folded plot."""
    phase = np.asarray(phase, float)
    mag = np.asarray(mag, float)
    m = np.isfinite(phase) & np.isfinite(mag)
    phase, mag = phase[m], mag[m]
    edges = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, nbins - 1)
    ph, mg, er, ct = [], [], [], []
    for b in range(nbins):
        sel = mag[idx == b]
        if sel.size == 0:
            continue
        ph.append(0.5 * (edges[b] + edges[b + 1]))
        mg.append(float(np.mean(sel)))
        er.append(float(np.std(sel, ddof=1) / np.sqrt(sel.size)) if sel.size > 1 else 0.0)
        ct.append(int(sel.size))
    return np.array(ph), np.array(mg), np.array(er), np.array(ct)
