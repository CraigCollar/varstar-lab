"""Server-side matplotlib figures, rendered to PNG bytes.

Every figure can be drawn on the app's dark background for screen use or on
white for pasting into a written report, hence the `theme` argument throughout.
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DARK = {
    "bg": "#0d1117", "panel": "#111823", "fg": "#e6edf3", "muted": "#8b98a9",
    "grid": "#22303f", "accent": "#4fb3ff", "accent2": "#ffb454",
    "good": "#3fd68a", "bad": "#ff6b6b", "purple": "#b98cff",
}
LIGHT = {
    "bg": "#ffffff", "panel": "#ffffff", "fg": "#14181f", "muted": "#5b6675",
    "grid": "#d7dde5", "accent": "#0b6bcb", "accent2": "#b8600a",
    "good": "#1a7f4f", "bad": "#c0392b", "purple": "#6b3fbf",
}


def palette(theme="dark"):
    return DARK if theme == "dark" else LIGHT


def _style(theme):
    c = palette(theme)
    plt.rcParams.update({
        "figure.facecolor": c["bg"], "axes.facecolor": c["bg"],
        "savefig.facecolor": c["bg"], "text.color": c["fg"],
        "axes.labelcolor": c["fg"], "axes.edgecolor": c["grid"],
        "xtick.color": c["muted"], "ytick.color": c["muted"],
        "grid.color": c["grid"], "grid.alpha": 0.55, "grid.linewidth": 0.6,
        "axes.grid": True, "axes.titlesize": 11.5, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "font.size": 10, "axes.linewidth": 0.9,
        "legend.framealpha": 0.85, "legend.facecolor": c["panel"],
        "legend.edgecolor": c["grid"], "figure.dpi": 110,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return c


def _out(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    return buf.getvalue()


def _fmt_t(t):
    """Return (hours since start, axis label, JD offset used)."""
    t = np.asarray(t, float)
    t0 = np.nanmin(t)
    return (t - t0) * 24.0, f"Hours since {t0:.5f}", t0


# --------------------------------------------------------------------------

def light_curve(t, mag, sigma=None, model_t=None, model_mag=None, theme="dark",
                title="Differential light curve", ylabel="Delta magnitude",
                calibrated=False, sessions=None):
    c = _style(theme)
    hrs, xlabel, t0 = _fmt_t(t)
    m = np.isfinite(hrs) & np.isfinite(mag)

    multi = sessions is not None and len(set(sessions)) > 1
    if multi:
        uniq = sorted(set(np.asarray(sessions)[m]))
        fig, axes = plt.subplots(1, len(uniq), figsize=(3.2 + 3.0 * len(uniq), 4.0),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, s in zip(axes, uniq):
            sel = m & (np.asarray(sessions) == s)
            h = hrs[sel] - np.nanmin(hrs[sel])
            if sigma is not None:
                ax.errorbar(h, np.asarray(mag)[sel], yerr=np.asarray(sigma)[sel],
                            fmt="o", ms=2.6, lw=0, elinewidth=0.6, capsize=0,
                            color=c["accent"], ecolor=c["grid"], alpha=0.95)
            else:
                ax.plot(h, np.asarray(mag)[sel], "o", ms=2.6, color=c["accent"])
            if model_t is not None:
                ms = np.asarray(sessions) == s
                mt, mm = np.asarray(model_t), np.asarray(model_mag)
                order = np.argsort(mt[ms])
                ax.plot((mt[ms][order] - np.nanmin(np.asarray(t)[sel])) * 24.0,
                        mm[ms][order], "-", lw=1.4, color=c["accent2"], alpha=0.9)
            ax.set_xlabel("Hours into session")
            ax.set_title(f"Session {int(s) + 1}", fontsize=10, color=c["muted"])
            ax.invert_yaxis()
        axes[0].set_ylabel(ylabel)
        fig.suptitle(title, fontsize=12, color=c["fg"], y=1.0)
        return _out(fig)

    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    if sigma is not None:
        ax.errorbar(hrs[m], np.asarray(mag)[m], yerr=np.asarray(sigma)[m],
                    fmt="o", ms=3.0, lw=0, elinewidth=0.7, capsize=0,
                    color=c["accent"], ecolor=c["grid"], alpha=0.95, label="measured")
    else:
        ax.plot(hrs[m], np.asarray(mag)[m], "o", ms=3.0, color=c["accent"],
                label="measured")
    if model_t is not None and model_mag is not None:
        mt = (np.asarray(model_t) - t0) * 24.0
        order = np.argsort(mt)
        ax.plot(mt[order], np.asarray(model_mag)[order], "-", lw=1.6,
                color=c["accent2"], label="Fourier model")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel + ("  (calibrated V)" if calibrated else ""))
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(loc="best")
    return _out(fig)


def periodogram(freq, power, best_freq=None, peaks=None, theme="dark",
                catalog_period=None, rayleigh=None, fap_levels=None,
                title="Lomb-Scargle periodogram"):
    c = _style(theme)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.4, 5.4),
                                  gridspec_kw={"height_ratios": [2.1, 1]})
    freq = np.asarray(freq, float)
    power = np.asarray(power, float)

    ax.plot(freq, power, lw=0.8, color=c["accent"])
    if best_freq:
        ax.axvline(best_freq, color=c["accent2"], lw=1.2, ls="-", alpha=0.9,
                   label=f"best f = {best_freq:.5f} c/d  (P = {1 / best_freq:.6f} d)")
    if catalog_period:
        ax.axvline(1.0 / catalog_period, color=c["purple"], lw=1.1, ls="--",
                   alpha=0.9, label=f"catalog P = {catalog_period:.7f} d")
    if peaks:
        for pk in peaks[1:]:
            ax.axvline(pk["freq"], color=c["muted"], lw=0.7, ls=":", alpha=0.7)
    if fap_levels:
        for lbl, lvl in fap_levels.items():
            if np.isfinite(lvl):
                ax.axhline(lvl, color=c["bad"], lw=0.7, ls="--", alpha=0.55)
                ax.text(freq[-1], lvl, f" FAP {lbl}", fontsize=7.5,
                        color=c["bad"], va="bottom", ha="right")
    ax.set_ylabel("Lomb-Scargle power")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(freq.min(), freq.max())

    # Zoom on the peak, showing the intrinsic peak width.
    if best_freq:
        w = (rayleigh or (freq[-1] - freq[0]) / 200) * 6
        lo, hi = best_freq - w, best_freq + w
        sel = (freq >= lo) & (freq <= hi)
        if sel.sum() > 4:
            ax2.plot(freq[sel], power[sel], lw=1.1, color=c["accent"])
            ax2.axvline(best_freq, color=c["accent2"], lw=1.2)
            if catalog_period and lo <= 1 / catalog_period <= hi:
                ax2.axvline(1 / catalog_period, color=c["purple"], lw=1.1, ls="--")
            if rayleigh:
                # Sit the width marker below the peak so it does not collide with
                # the vertical best-frequency line or its label.
                ymark = power[sel].max() * 0.30
                ax2.annotate("", xy=(best_freq - rayleigh / 2, ymark),
                             xytext=(best_freq + rayleigh / 2, ymark),
                             arrowprops=dict(arrowstyle="<->", color=c["fg"], lw=1.1))
                ax2.text(best_freq, ymark + power[sel].max() * 0.055,
                         f"1/T = {rayleigh:.3f} c/d", ha="center", fontsize=8,
                         color=c["fg"],
                         bbox=dict(boxstyle="round,pad=0.22", fc=c["bg"],
                                   ec="none", alpha=0.85))
            ax2.set_xlim(lo, hi)
    ax2.set_xlabel("Frequency (cycles per day)")
    ax2.set_ylabel("power")
    ax2.set_title("Peak detail", fontsize=9.5, color=c["muted"])
    fig.tight_layout()
    return _out(fig)


def pdm_plot(period, theta, best=None, catalog_period=None, theme="dark"):
    c = _style(theme)
    fig, ax = plt.subplots(figsize=(9.4, 3.4))
    ax.plot(np.asarray(period) * 24.0, theta, lw=0.8, color=c["good"])
    if best:
        ax.axvline(best * 24.0, color=c["accent2"], lw=1.2,
                   label=f"theta minimum at P = {best:.6f} d ({best * 24:.4f} h)")
    if catalog_period:
        ax.axvline(catalog_period * 24.0, color=c["purple"], lw=1.0, ls="--",
                   label=f"catalog P = {catalog_period * 24:.4f} h")
    ax.axhline(1.0, color=c["muted"], lw=0.7, ls=":")
    ax.set_xlabel("Trial period (hours)")
    ax.set_ylabel(r"PDM $\Theta$")
    ax.set_title("Phase dispersion minimisation - an estimator that assumes no curve shape")
    ax.legend(loc="best", fontsize=8)
    return _out(fig)


def folded(phase, mag, sigma=None, binned=None, model_phase=None, model_mag=None,
           period=None, theme="dark", ylabel="Delta magnitude", repeat=True,
           title=None):
    c = _style(theme)
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    phase = np.asarray(phase, float)
    mag = np.asarray(mag, float)
    m = np.isfinite(phase) & np.isfinite(mag)

    shifts = [0.0, 1.0] if repeat else [0.0]
    for k, sh in enumerate(shifts):
        lbl = "measured" if k == 0 else None
        if sigma is not None:
            ax.errorbar(phase[m] + sh, mag[m], yerr=np.asarray(sigma)[m], fmt="o",
                        ms=2.8, lw=0, elinewidth=0.5, capsize=0, color=c["accent"],
                        ecolor=c["grid"], alpha=0.55, label=lbl)
        else:
            ax.plot(phase[m] + sh, mag[m], "o", ms=2.8, color=c["accent"],
                    alpha=0.55, label=lbl)

    if binned is not None:
        bp, bm, be, _ = binned
        for k, sh in enumerate(shifts):
            ax.errorbar(np.asarray(bp) + sh, bm, yerr=be, fmt="s", ms=5.0,
                        color=c["accent2"], ecolor=c["accent2"], elinewidth=1.2,
                        capsize=2.5, lw=0, label="phase-binned mean" if k == 0 else None,
                        zorder=5)

    if model_phase is not None and model_mag is not None:
        mp = np.asarray(model_phase, float)
        mm = np.asarray(model_mag, float)
        order = np.argsort(mp)
        for k, sh in enumerate(shifts):
            ax.plot(mp[order] + sh, mm[order], "-", lw=1.8, color=c["good"],
                    label="Fourier model" if k == 0 else None, zorder=6)

    if repeat:
        ax.axvline(1.0, color=c["grid"], lw=0.8, ls="-")
    ax.set_xlabel("Pulsation phase" + (" (two cycles shown)" if repeat else ""))
    ax.set_ylabel(ylabel)
    ax.set_title(title or (f"Phase-folded at P = {period:.6f} d = {period * 24:.4f} h"
                           if period else "Phase-folded light curve"))
    ax.invert_yaxis()
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0, 2 if repeat else 1)
    return _out(fig)


def _break_gaps(x_hours, y, gap_hours=1.0):
    """Insert NaN across observing gaps so lines are not drawn through them.

    Without this, a three-night run is joined by long diagonal segments that
    look like real measurements spanning the daytime.
    """
    x = np.asarray(x_hours, float)
    y = np.asarray(y, float)
    if x.size < 2:
        return x, y
    out_x, out_y = [x[0]], [y[0]]
    for i in range(1, x.size):
        if x[i] - x[i - 1] > gap_hours:
            out_x.append(np.nan)
            out_y.append(np.nan)
        out_x.append(x[i])
        out_y.append(y[i])
    return np.asarray(out_x), np.asarray(out_y)


def diagnostics(t, fwhm, sky, xpos, ypos, flux_comp, theme="dark"):
    """Observing-condition panels - where a suspicious light curve gets explained."""
    c = _style(theme)
    hrs, xlabel, _ = _fmt_t(t)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6))

    def line(ax, y, color, label=None):
        gx, gy = _break_gaps(hrs, y)
        ax.plot(gx, gy, "-", lw=1.0, color=color, label=label)

    ax = axes[0, 0]
    line(ax, fwhm, c["accent"])
    ax.set_ylabel("FWHM (pixels)")
    ax.set_title("Seeing", fontsize=10)

    ax = axes[0, 1]
    med = np.nanmedian(flux_comp) or 1.0
    line(ax, np.asarray(flux_comp) / med, c["accent2"])
    ax.set_ylabel("relative")
    ax.set_title("Comparison-ensemble flux (transparency)", fontsize=10)

    ax = axes[1, 0]
    x0 = np.asarray(xpos)[:, 0] if np.asarray(xpos).ndim > 1 else np.asarray(xpos)
    y0 = np.asarray(ypos)[:, 0] if np.asarray(ypos).ndim > 1 else np.asarray(ypos)
    line(ax, x0 - x0[0], c["good"], "dx")
    line(ax, y0 - y0[0], c["purple"], "dy")
    ax.set_ylabel("drift (pixels)")
    ax.set_xlabel(xlabel)
    ax.set_title("Target drift on the sensor", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    s = np.asarray(sky)
    s = s[:, 0] if s.ndim > 1 else s
    line(ax, s, c["muted"])
    ax.set_ylabel("sky (ADU/pixel)")
    ax.set_xlabel(xlabel)
    ax.set_title("Sky background", fontsize=10)

    fig.tight_layout()
    return _out(fig)


def field_chart(preview_shape, stars, target=None, comps=None, theme="dark",
                r_ap=6.0, scale=1.0):
    """Vector overlay of the selected apertures - a finder chart for the report."""
    c = _style(theme)
    h, w = preview_shape
    fig, ax = plt.subplots(figsize=(6.4, 6.4 * h / max(1, w)))
    ax.set_facecolor(c["bg"])
    ax.grid(False)
    for i, s in enumerate(stars):
        x, y = s["x"] * scale, s["y"] * scale
        if target is not None and i == target:
            col, lab, rr = c["bad"], "T", r_ap * 1.5
        elif comps and i in comps:
            col, lab, rr = c["good"], f"C{comps.index(i) + 1}", r_ap * 1.2
        else:
            col, lab, rr = c["muted"], "", r_ap
        ax.add_patch(plt.Circle((x, y), rr, fill=False, color=col, lw=1.3))
        if lab:
            ax.text(x + rr * 1.4, y, lab, color=col, fontsize=9, va="center")
    ax.set_xlim(0, w * scale)
    ax.set_ylim(h * scale, 0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Aperture layout", fontsize=10)
    return _out(fig)


def distance_summary(result, gaia=None, theme="dark"):
    """Error budget and the distance interval, side by side."""
    c = _style(theme)
    dist = result["distance"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    budget = dist["error_budget"]
    labels = ["mean magnitude", "P-L relation\n(incl. period)", "extinction"]
    vals = [budget["mean_magnitude"], budget["absolute_magnitude"],
            budget["extinction"]]
    colors = [c["accent"], c["accent2"], c["purple"]]
    y = np.arange(len(labels))
    ax1.barh(y, vals, color=colors, height=0.55)
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.set_xlabel("contribution to sigma(mu)  [mag]")
    ax1.set_title(f"Error budget - total {budget['total_modulus']:.3f} mag",
                  fontsize=10)
    ax1.invert_yaxis()
    for yi, v in zip(y, vals):
        if np.isfinite(v):
            ax1.text(v, yi, f" {v:.3f}", va="center", fontsize=8, color=c["fg"])

    entries = [("This work\n(period-luminosity)", dist["distance_pc"],
                dist["distance_pc_lo"], dist["distance_pc_hi"], c["accent"])]
    if gaia and gaia.get("ok") and gaia.get("distance_pc"):
        entries.append(("Gaia DR3\nparallax", gaia["distance_pc"],
                        gaia.get("distance_pc_lo", gaia["distance_pc"]),
                        min(gaia.get("distance_pc_hi", gaia["distance_pc"]),
                            gaia["distance_pc"] * 4),
                        c["good"]))
    ypos = np.arange(len(entries))
    for yi, (lab, d, lo, hi, col) in zip(ypos, entries):
        ax2.errorbar([d], [yi], xerr=[[max(0, d - lo)], [max(0, hi - d)]],
                     fmt="o", ms=9, color=col, ecolor=col, elinewidth=2.4,
                     capsize=6)
        ax2.text(d, yi + 0.22, f"{d:,.0f} pc = {d * 3.2616:,.0f} ly",
                 ha="center", fontsize=8.5, color=c["fg"])
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([e[0] for e in entries], fontsize=8.5)
    ax2.set_ylim(-0.7, len(entries) - 0.3)
    ax2.set_xlabel("Distance (parsecs)")
    ax2.set_title("Distance to the star", fontsize=10)

    fig.tight_layout()
    return _out(fig)


def pl_relation(result, theme="dark", relations=None):
    """Where this star lands on the period-luminosity plane."""
    from .distance import PL_RELATIONS

    c = _style(theme)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    p = np.logspace(np.log10(0.02), np.log10(0.30), 200)
    logp = np.log10(p)
    cols = [c["accent"], c["accent2"], c["purple"]]
    for (key, rel), col in zip(PL_RELATIONS.items(), cols):
        mv = rel["slope"] * logp + rel["intercept"]
        lw = 2.0 if key == result["absolute"]["relation_key"] else 1.0
        al = 1.0 if key == result["absolute"]["relation_key"] else 0.5
        ax.plot(p * 24, mv, "-", lw=lw, alpha=al, color=col,
                label=rel["label"].split(" - ")[0])
        if key == result["absolute"]["relation_key"]:
            ax.fill_between(p * 24, mv - rel["scatter"], mv + rel["scatter"],
                            color=col, alpha=0.14)

    per = result["period"]
    ax.errorbar([per * 24], [result["absolute"]["M"]],
                yerr=[result["absolute"]["sigma_M"]],
                xerr=[[result["sigma_period"] * 24], [result["sigma_period"] * 24]],
                fmt="*", ms=17, color=c["good"], ecolor=c["good"], elinewidth=1.5,
                capsize=4, label="this star", zorder=6)
    ax.set_xscale("log")
    ax.set_xlabel("Period (hours)")
    ax.set_ylabel(r"Absolute magnitude $M_V$")
    ax.invert_yaxis()
    ax.set_title("Delta Scuti period-luminosity relation")
    ax.legend(loc="best", fontsize=8)
    return _out(fig)


def bootstrap_hist(periods, best=None, theme="dark"):
    c = _style(theme)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    arr = np.asarray(periods, float) * 24.0
    ax.hist(arr, bins=min(40, max(10, len(arr) // 8)), color=c["accent"],
            alpha=0.85, edgecolor=c["bg"])
    if best:
        ax.axvline(best * 24.0, color=c["accent2"], lw=1.6, label="best fit")
        ax.legend(fontsize=8)
    ax.set_xlabel("Period (hours)")
    ax.set_ylabel("bootstrap trials")
    ax.set_title("Bootstrap period distribution", fontsize=10)
    return _out(fig)
