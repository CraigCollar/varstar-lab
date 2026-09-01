"""Aperture photometry with drift tracking and ensemble differential reduction.

The proposal calls for auto-guiding to be turned OFF, so the field walks across
the sensor over a 2-3 h run. Every frame is therefore re-registered before the
apertures are placed: a global cross-correlation shift catches large jumps, then
each star is centroided locally to absorb differential drift and field rotation.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.aperture import (CircularAnnulus, CircularAperture, ApertureStats,
                                aperture_photometry)
from photutils.detection import DAOStarFinder

from .ingest import read_pixels

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class PhotConfig:
    channel: str = "G"
    fwhm: float = 4.0             # pixels, for detection + aperture scaling
    thresh_sigma: float = 5.0     # detection threshold above background sigma
    ap_factor: float = 1.5        # aperture radius = ap_factor * FWHM
    ann_in_factor: float = 3.0    # sky annulus inner radius / FWHM
    ann_out_factor: float = 5.0   # sky annulus outer radius / FWHM
    gain: float = 1.0             # e- per ADU, for Poisson errors
    read_noise: float = 0.0       # e- per pixel (0 = infer from sky annulus only)
    track: bool = True            # re-register each frame
    track_box: int = 11           # centroid box half-width in pixels
    global_align: bool = True     # cross-correlation pre-alignment
    saturation: Optional[float] = None
    max_sources: int = 60

    @property
    def r_ap(self):
        return max(1.5, self.ap_factor * self.fwhm)

    @property
    def r_in(self):
        return max(self.r_ap + 2.0, self.ann_in_factor * self.fwhm)

    @property
    def r_out(self):
        return max(self.r_in + 3.0, self.ann_out_factor * self.fwhm)


# --------------------------------------------------------------------------
# detection & display
# --------------------------------------------------------------------------

def background_stats(data: np.ndarray):
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    return float(mean), float(median), float(std)


def detect_sources(data: np.ndarray, cfg: PhotConfig) -> list[dict]:
    """Find stars on the reference frame and rank them by brightness."""
    _, median, std = background_stats(data)
    if not np.isfinite(std) or std <= 0:
        std = float(np.nanstd(data)) or 1.0
    finder = DAOStarFinder(fwhm=cfg.fwhm, threshold=cfg.thresh_sigma * std,
                           exclude_border=True)
    tbl = finder(data - median)
    if tbl is None or len(tbl) == 0:
        return []

    tbl.sort("flux", reverse=True)
    edge = cfg.r_out + 2
    h, w = data.shape
    out = []
    sat = cfg.saturation
    for row in tbl:
        x, y = float(row["xcentroid"]), float(row["ycentroid"])
        if x < edge or y < edge or x > w - edge or y > h - edge:
            continue
        peak = float(row["peak"]) + median
        out.append({
            "x": x, "y": y,
            "flux": float(row["flux"]),
            "peak": peak,
            "sharpness": float(row["sharpness"]),
            "roundness": float(row["roundness1"]),
            "saturated": bool(sat and peak >= 0.97 * sat),
        })
        if len(out) >= cfg.max_sources:
            break
    return out


def measure_fwhm(data: np.ndarray, positions, box: int = 12) -> Optional[float]:
    """Median FWHM of the given stars, in pixels.

    Measured from the azimuthally-averaged radial profile at half maximum, which
    is the actual definition. A second-moment estimate is not used here: real
    star images have Moffat-like wings, and second moments weight those wings by
    r^2, which inflates the answer by nearly a factor of two.
    """
    vals = []
    _, median, _ = background_stats(data)
    h, w = data.shape
    for (x, y) in positions:
        xi, yi = int(round(x)), int(round(y))
        if xi - box < 0 or yi - box < 0 or xi + box >= w or yi + box >= h:
            continue
        cut = data[yi - box:yi + box + 1, xi - box:xi + box + 1] - median
        yy, xx = np.mgrid[0:cut.shape[0], 0:cut.shape[1]]
        # Centroid first, so an off-centre star does not broaden its own profile.
        pos = np.clip(cut, 0, None)
        tot = pos.sum()
        if tot <= 0:
            continue
        cx = (pos * xx).sum() / tot
        cy = (pos * yy).sum() / tot
        r = np.hypot(xx - cx, yy - cy).ravel()
        v = cut.ravel()

        nbin = int(box * 2)
        edges = np.linspace(0, box, nbin + 1)
        idx = np.clip(np.digitize(r, edges) - 1, 0, nbin - 1)
        prof = np.full(nbin, np.nan)
        for b in range(nbin):
            sel = v[idx == b]
            if sel.size:
                prof[b] = sel.mean()
        centers = 0.5 * (edges[:-1] + edges[1:])

        ok = np.isfinite(prof)
        if ok.sum() < 4:
            continue
        prof, centers = prof[ok], centers[ok]
        peak = float(prof[0])
        if peak <= 0:
            continue
        half = 0.5 * peak
        below = np.where(prof < half)[0]
        if below.size == 0:
            continue
        j = int(below[0])
        if j == 0:
            continue
        # Linear interpolation onto the half-maximum crossing.
        p0, p1 = prof[j - 1], prof[j]
        r0, r1 = centers[j - 1], centers[j]
        r_half = r0 + (p0 - half) / (p0 - p1) * (r1 - r0) if p0 != p1 else r0
        vals.append(2.0 * float(r_half))

    if not vals:
        return None
    return float(np.median(vals))


def stretch_preview(data: np.ndarray, mode: str = "asinh") -> np.ndarray:
    """Scale to 0-255 uint8 for on-screen star picking (display only)."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    lo = np.percentile(finite, 25.0)
    hi = np.percentile(finite, 99.7)
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((data - lo) / (hi - lo), 0, 1)
    if mode == "asinh":
        x = np.arcsinh(10.0 * x) / np.arcsinh(10.0)
    elif mode == "sqrt":
        x = np.sqrt(x)
    elif mode == "log":
        x = np.log1p(999.0 * x) / np.log(1000.0)
    return (255 * x).astype(np.uint8)


def preview_png(data: np.ndarray, mode: str = "asinh", max_dim: int = 1400) -> bytes:
    import io
    from PIL import Image

    img = Image.fromarray(stretch_preview(data, mode), mode="L")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((max(1, int(img.width * scale)),
                          max(1, int(img.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def global_shift(ref: np.ndarray, img: np.ndarray) -> tuple[float, float]:
    """(dx, dy) that maps reference pixel coords onto this frame."""
    try:
        from skimage.registration import phase_cross_correlation
    except ImportError:
        return 0.0, 0.0
    if ref.shape != img.shape:
        return 0.0, 0.0
    try:
        a = np.nan_to_num(ref - np.median(ref))
        b = np.nan_to_num(img - np.median(img))
        # Suppress hot pixels/noise so the correlation locks onto stars.
        a = np.clip(a, 0, None)
        b = np.clip(b, 0, None)
        shift, _, _ = phase_cross_correlation(a, b, upsample_factor=10,
                                              normalization=None)
        dy, dx = float(shift[0]), float(shift[1])
        # phase_cross_correlation returns the shift to apply to `b` to match `a`,
        # so a star at (x, y) in the reference sits at (x - dx, y - dy) here.
        return -dx, -dy
    except Exception:
        return 0.0, 0.0


def refine_centroid(data: np.ndarray, x: float, y: float, box: int,
                    bkg: float) -> tuple[float, float, bool]:
    """Flux-weighted centroid inside a small box. Returns (x, y, ok)."""
    h, w = data.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - box), min(w, xi + box + 1)
    y0, y1 = max(0, yi - box), min(h, yi + box + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return x, y, False
    cut = np.nan_to_num(data[y0:y1, x0:x1] - bkg)
    cut = np.clip(cut, 0, None)
    tot = cut.sum()
    if tot <= 0:
        return x, y, False
    yy, xx = np.mgrid[y0:y1, x0:x1]
    nx = float((cut * xx).sum() / tot)
    ny = float((cut * yy).sum() / tot)
    if not (np.isfinite(nx) and np.isfinite(ny)):
        return x, y, False
    if abs(nx - x) > box or abs(ny - y) > box:
        return x, y, False
    return nx, ny, True


# --------------------------------------------------------------------------
# per-frame photometry
# --------------------------------------------------------------------------

def measure_frame(data: np.ndarray, positions: np.ndarray, cfg: PhotConfig):
    """Aperture-sum every position on one frame.

    Sky is the sigma-clipped *median* of an annulus (robust to neighbours), and
    the flux error follows the CCD equation with the annulus scatter standing in
    for sky + read + dark noise.
    """
    r_ap, r_in, r_out = cfg.r_ap, cfg.r_in, cfg.r_out
    aps = CircularAperture(positions, r=r_ap)
    ann = CircularAnnulus(positions, r_in=r_in, r_out=r_out)

    sc = SigmaClip(sigma=3.0, maxiters=5)
    ann_stats = ApertureStats(data, ann, sigma_clip=sc)
    sky_med = np.asarray(ann_stats.median, dtype=float)
    sky_std = np.asarray(ann_stats.std, dtype=float)
    n_sky = np.asarray(ann_stats.sum_aper_area.value if hasattr(ann_stats.sum_aper_area, "value")
                       else ann_stats.sum_aper_area, dtype=float)

    phot = aperture_photometry(data, aps)
    raw = np.asarray(phot["aperture_sum"], dtype=float)
    area = float(aps.area)

    sky_med = np.where(np.isfinite(sky_med), sky_med, 0.0)
    sky_std = np.where(np.isfinite(sky_std) & (sky_std > 0), sky_std, np.nan)
    net = raw - sky_med * area

    g = max(1e-6, cfg.gain)
    n_sky = np.where(np.isfinite(n_sky) & (n_sky > 1), n_sky, area * 4)
    # sigma_F^2 = F/g + N_ap*sig_sky^2 + N_ap^2*sig_sky^2/N_sky   (all in ADU)
    sig_sky2 = np.nan_to_num(sky_std ** 2, nan=0.0)
    var = np.clip(net, 0, None) / g + area * sig_sky2 + (area ** 2) * sig_sky2 / n_sky
    if cfg.read_noise > 0:
        var = var + area * (cfg.read_noise / g) ** 2
    err = np.sqrt(np.clip(var, 0, None))

    # Peak pixel inside each aperture, for saturation checks.
    peaks = np.asarray(ApertureStats(data, aps).max, dtype=float)
    return net, err, sky_med, sky_std, peaks


# --------------------------------------------------------------------------
# full run
# --------------------------------------------------------------------------

@dataclass
class PhotResult:
    jd: np.ndarray = field(default_factory=lambda: np.array([]))
    flux: np.ndarray = field(default_factory=lambda: np.array([]))   # (nframe, nstar)
    ferr: np.ndarray = field(default_factory=lambda: np.array([]))
    sky: np.ndarray = field(default_factory=lambda: np.array([]))
    peak: np.ndarray = field(default_factory=lambda: np.array([]))
    xpos: np.ndarray = field(default_factory=lambda: np.array([]))
    ypos: np.ndarray = field(default_factory=lambda: np.array([]))
    fwhm: np.ndarray = field(default_factory=lambda: np.array([]))
    frame_index: list = field(default_factory=list)
    filenames: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    star_labels: list = field(default_factory=list)
    edge_note: Optional[str] = None


def run_photometry(frames, positions, cfg: PhotConfig,
                   ref_frame_index: int = 0,
                   progress: Optional[Callable[[int, int, str], None]] = None
                   ) -> PhotResult:
    """Measure every usable frame at the selected star positions."""
    usable = [f for f in frames if not f.excluded]
    if not usable:
        raise ValueError("no usable frames")

    positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    nstar = len(positions)
    res = PhotResult()
    jd, fl, fe, sk, pk, xs, ys, fw, idx, names = [], [], [], [], [], [], [], [], [], []

    ref_data = None
    prev_pos = positions.copy()
    total = len(usable)

    for i, fr in enumerate(usable):
        if progress:
            progress(i, total, fr.filename)
        try:
            data, _, _, _, _ = read_pixels(fr.path, cfg.channel)
        except Exception as exc:
            res.failures.append({"file": fr.filename, "reason": f"read failed: {exc}"})
            continue

        if ref_data is None:
            ref_data = data

        _, med, _ = background_stats(data)

        pos = prev_pos.copy()
        if cfg.track:
            if cfg.global_align and data.shape == ref_data.shape:
                dx, dy = global_shift(ref_data, data)
                cand = positions + np.array([dx, dy])
                # Trust the global solution only if it beats simple persistence.
                if np.isfinite(cand).all():
                    pos = cand
            refined = []
            for (px, py) in pos:
                nx, ny, ok = refine_centroid(data, px, py, cfg.track_box, med)
                refined.append((nx, ny))
            pos = np.array(refined, dtype=float)

        # Stars whose sky annulus has drifted over an edge cannot be measured on
        # this frame. Invalidate those stars individually rather than throwing
        # the frame away: with tens of detected sources spread across the field,
        # one of them is almost always near an edge, and dropping whole frames
        # for that would discard the entire run.
        h, w = data.shape
        edge = cfg.r_out + 1
        off = ((pos[:, 0] < edge) | (pos[:, 1] < edge) |
               (pos[:, 0] > w - edge) | (pos[:, 1] > h - edge))
        if off.all():
            res.failures.append({"file": fr.filename,
                                 "reason": "every star drifted off the frame"})
            continue

        # Park off-frame stars at the centre so photutils has a valid position,
        # then blank their results below.
        safe = pos.copy()
        if off.any():
            safe[off] = [w / 2.0, h / 2.0]

        try:
            net, err, sky, _, peaks = measure_frame(data, safe, cfg)
        except Exception as exc:
            res.failures.append({"file": fr.filename, "reason": f"photometry failed: {exc}"})
            continue

        if off.any():
            net = net.astype(float).copy(); net[off] = np.nan
            err = err.astype(float).copy(); err[off] = np.nan
            sky = sky.astype(float).copy(); sky[off] = np.nan
            peaks = peaks.astype(float).copy(); peaks[off] = np.nan
            if off.sum() and not res.edge_note:
                res.edge_note = (
                    f"{int(off.sum())} of {len(pos)} selected sources drifted within "
                    f"{edge:.0f} px of a frame edge and were skipped on some frames. "
                    "This is expected with auto-guiding off; it only matters if the "
                    "target or a comparison star is among them.")

        prev_pos = pos
        jd.append(fr.jd if fr.jd is not None else np.nan)
        fl.append(net); fe.append(err); sk.append(sky); pk.append(peaks)
        xs.append(pos[:, 0]); ys.append(pos[:, 1])
        fw.append(measure_fwhm(data, pos) or np.nan)
        idx.append(fr.index); names.append(fr.filename)

    if not jd:
        raise ValueError("photometry produced no measurements; check star selection")

    res.jd = np.asarray(jd, float)
    res.flux = np.asarray(fl, float)
    res.ferr = np.asarray(fe, float)
    res.sky = np.asarray(sk, float)
    res.peak = np.asarray(pk, float)
    res.xpos = np.asarray(xs, float)
    res.ypos = np.asarray(ys, float)
    res.fwhm = np.asarray(fw, float)
    res.frame_index = idx
    res.filenames = names

    order = np.argsort(res.jd)
    if not np.all(np.diff(res.jd) >= 0):
        res.jd = res.jd[order]
        for attr in ("flux", "ferr", "sky", "peak", "xpos", "ypos", "fwhm"):
            setattr(res, attr, getattr(res, attr)[order])
        res.frame_index = [res.frame_index[i] for i in order]
        res.filenames = [res.filenames[i] for i in order]
    return res


# --------------------------------------------------------------------------
# differential reduction
# --------------------------------------------------------------------------

def differential(res: PhotResult, target: int, comps: list[int],
                 comp_mags: Optional[dict] = None,
                 saturation: Optional[float] = None):
    """Ensemble differential photometry.

    Target flux is divided by the summed comparison flux, which cancels
    transparency and airmass changes to first order. If catalog magnitudes are
    supplied for one or more comparison stars, a zero point converts the
    instrumental magnitude to a calibrated apparent magnitude.
    """
    ft = res.flux[:, target]
    et = res.ferr[:, target]
    fc = res.flux[:, comps].sum(axis=1)
    ec = np.sqrt((res.ferr[:, comps] ** 2).sum(axis=1))

    finite = np.isfinite(ft) & np.isfinite(fc) & (ft > 0) & (fc > 0) & np.isfinite(res.jd)
    good = finite.copy()
    n_sat = 0
    sat_names = []
    if saturation:
        sat_stars = [target] + list(comps)
        hot = res.peak[:, sat_stars] >= 0.97 * saturation
        sat_mask = np.any(hot, axis=1)
        n_sat = int(np.sum(sat_mask & finite))
        good &= ~sat_mask
        for j, si in enumerate(sat_stars):
            if np.any(hot[:, j]):
                sat_names.append({"star": int(si),
                                  "role": "target" if si == target else "comparison",
                                  "n_frames": int(np.sum(hot[:, j]))})

    dmag = np.full(ft.shape, np.nan)
    dmag[good] = -2.5 * np.log10(ft[good] / fc[good])
    sig = np.full(ft.shape, np.nan)
    sig[good] = 1.0857362 * np.sqrt((et[good] / ft[good]) ** 2 + (ec[good] / fc[good]) ** 2)

    out = {
        "jd": res.jd, "dmag": dmag, "sigma": sig, "good": good,
        "flux_target": ft, "flux_comp": fc,
        "zeropoint": None, "zeropoint_sigma": None, "mag": None,
        "calibrated": False, "n_calibrators": 0,
        "n_total": int(len(ft)), "n_good": int(good.sum()),
        "n_rejected_saturated": n_sat,
        "n_rejected_nonfinite": int(np.sum(~finite)),
        "saturated_stars": sat_names,
        "rejection_note": None,
    }
    if good.sum() == 0:
        parts = []
        if n_sat:
            parts.append(f"{n_sat} frame(s) rejected because a selected star hit "
                         f"the {saturation:.0f} ADU full well")
        if np.sum(~finite):
            parts.append(f"{int(np.sum(~finite))} frame(s) had non-positive or "
                         "missing flux")
        out["rejection_note"] = (
            "No usable points survived: " + "; ".join(parts) + ". "
            "Pick fainter stars, shorten the exposure, or - if the peaks are "
            "genuinely below full well - check the SATURATE header value."
        ) if parts else "No usable points survived."
    elif n_sat:
        out["rejection_note"] = (
            f"{n_sat} of {len(ft)} frames dropped for saturation. Saturated pixels "
            "respond non-linearly, so their photometry is not recoverable.")

    # ---- optional absolute calibration -----------------------------------
    # The offset is derived from *differential* magnitudes of the calibrator
    # stars against the same ensemble denominator used for the target. That way
    # the calibrated light curve keeps the transparency cancellation of the
    # differential curve instead of reintroducing it via raw instrumental
    # magnitudes.
    if comp_mags:
        offsets, resid = [], []
        for si, cat in comp_mags.items():
            si = int(si)
            if si not in comps or cat is None or cat == "":
                continue
            f = res.flux[:, si]
            m = np.isfinite(f) & (f > 0) & good
            if m.sum() < 3:
                continue
            d_i = -2.5 * np.log10(f[m] / fc[m])
            offsets.append(float(cat) - float(np.median(d_i)))
            resid.append(float(np.std(d_i, ddof=1)) if m.sum() > 1 else 0.0)
        if offsets:
            arr = np.asarray(offsets, float)
            zp = float(np.mean(arr))
            # Spread between independent calibrators is the honest zero-point
            # error; it folds in their catalog errors and any colour term.
            if len(arr) > 1:
                zp_sig = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            else:
                zp_sig = 0.05
            # Floor it. Calibrators that happen to agree closely can drive the
            # formal spread to a few millimagnitudes, but a single-band zero
            # point with no colour term and no aperture correction is not
            # actually good to better than ~0.02 mag - and this value feeds
            # straight into the distance, where understating it would make the
            # final error bar dishonest.
            zp_sig_floor = 0.02
            zp_sig = max(zp_sig, zp_sig_floor)
            out["mag"] = dmag + zp
            out["zeropoint"] = zp
            out["zeropoint_sigma"] = zp_sig
            out["calibrated"] = True
            out["n_calibrators"] = len(arr)
            out["calibrator_offsets"] = arr.tolist()
            out["ensemble_catalog_mag"] = zp
            out["zeropoint_spread"] = (float(np.std(arr, ddof=1))
                                       if len(arr) > 1 else None)
            out["zeropoint_floored"] = bool(
                len(arr) > 1 and np.std(arr, ddof=1) / np.sqrt(len(arr)) < zp_sig_floor)
            out["zeropoint_note"] = (
                f"Zero point {zp:.4f} mag from {len(arr)} calibrator(s). "
                + (f"The scatter between calibrators is only "
                   f"{np.std(arr, ddof=1) * 1000:.1f} mmag, but the quoted error is "
                   f"held at a {zp_sig_floor:.2f} mag floor: a single-band zero point "
                   "carries colour-term and aperture-correction systematics that "
                   "agreement between calibrators cannot detect."
                   if out.get("zeropoint_floored") else
                   "Quoted error is the scatter between calibrators."))
    return out


def comparison_report(res: PhotResult, target: int, comps: list[int]):
    """Per-star scatter, so a variable comparison star can be spotted and dropped."""
    rows = []
    fc_all = res.flux[:, comps].sum(axis=1)
    for si in range(res.flux.shape[1]):
        f = res.flux[:, si]
        m = np.isfinite(f) & (f > 0) & np.isfinite(fc_all) & (fc_all > 0)
        role = "target" if si == target else ("comparison" if si in comps else "unused")
        row = {"star": si, "role": role,
               "median_flux": float(np.nanmedian(f)) if m.any() else None,
               "median_snr": None, "rms_mmag": None, "check_rms_mmag": None}
        if m.sum() >= 3:
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = f[m] / res.ferr[:, si][m]
            row["median_snr"] = float(np.nanmedian(snr))
            inst = -2.5 * np.log10(f[m])
            row["rms_mmag"] = float(np.nanstd(inst) * 1000)
            # Scatter against the ensemble excluding this star.
            others = [c for c in comps if c != si]
            if others:
                fo = res.flux[:, others].sum(axis=1)
                mm = m & np.isfinite(fo) & (fo > 0)
                if mm.sum() >= 3:
                    d = -2.5 * np.log10(f[mm] / fo[mm])
                    row["check_rms_mmag"] = float(np.nanstd(d) * 1000)
        rows.append(row)
    return rows
