"""Synthetic observing runs, for testing and teaching.

Generates FITS frames that imitate the run the proposal describes: unstacked
30-45 s exposures, 0 s delay, 2-3 h continuous, auto-guiding OFF. The simulated
defects are the ones that actually bite real photometry - field drift, seeing
changes, transparency variations, Poisson and read noise - so a reduction that
recovers the injected period here is doing real work.

The injected star is modelled on V0756 CrA: HADS(B), V = 11.43-12.0,
P_0 = 0.1071934 d, with a first-overtone mode at P_1/P_0 = 0.774.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from astropy.time import Time

from .timing import TARGET_DEFAULT

# Fourier template for a high-amplitude delta Scuti: fast rise, slow decline.
# Amplitude ratios R21, R31 and the phase difference phi21 are in the range
# measured for real HADS stars.
_HADS_HARMONICS = [(1.0, 0.0), (0.26, 4.05), (0.095, 1.95), (0.035, 5.9)]

# Southern site - V0756 CrA sits at dec -42, so a southern telescope is implied.
DEFAULT_SITE = {"name": "Siding Spring Observatory", "lat": -31.2733,
                "lon": 149.0644, "elev": 1165.0}


@dataclass
class SynthConfig:
    n_frames: int = 200
    exptime: float = 40.0          # seconds
    cadence: float = 45.0          # seconds (exposure + readout, 0 s delay)
    size: int = 600                # pixels per side
    n_field_stars: int = 17
    period: float = TARGET_DEFAULT["period_cat"]
    amplitude: float = 0.285       # semi-amplitude in magnitudes (0.57 peak-to-peak)
    mean_mag: float = 11.715       # midpoint of the catalogued 11.43-12.0 range
    second_mode: bool = True       # HADS(B) => double-mode
    ratio_p1_p0: float = 0.774
    amp_ratio_1: float = 0.20      # overtone amplitude / fundamental amplitude
    fwhm: float = 3.6              # pixels
    seeing_variation: float = 0.55 # peak-to-peak pixels
    drift_x: float = 13.0          # pixels per hour (no auto-guiding)
    drift_y: float = 7.5
    jitter: float = 0.45           # pixels rms, frame to frame
    sky: float = 320.0             # ADU per pixel
    sky_gradient: float = 0.06     # fractional across the frame
    gain: float = 1.5              # e- per ADU
    read_noise: float = 8.0        # e-
    # flux_ADU = 10^(0.4*(ZP - mag)) * exptime. Chosen so the target peaks near
    # 16 kADU and the brightest comparison near 33 kADU - well clear of the
    # 65535 ADU full well even when the seeing sharpens.
    zeropoint: float = 21.70
    transparency_rms: float = 0.02
    cloud_events: int = 1
    n_nights: int = 1
    night_gap_days: float = 1.0
    start_utc: str = "2026-07-14T14:10:00"   # ~midnight local at the site
    saturation: float = 65535.0
    seed: int = 20260714
    noise: bool = True
    site: dict = field(default_factory=lambda: dict(DEFAULT_SITE))


def _template(phase):
    """Sum of the HADS harmonics; larger value = brighter."""
    phase = np.asarray(phase, dtype=float)
    total = np.zeros_like(phase, dtype=float)
    for k, (amp, ph) in enumerate(_HADS_HARMONICS, start=1):
        total += amp * np.sin(2 * np.pi * k * phase + ph)
    return total


# Semi-amplitude of the raw template, so it can be rescaled to a requested one.
_ref = _template(np.linspace(0.0, 1.0, 8192, endpoint=False))
_TEMPLATE_SEMI = 0.5 * float(_ref.max() - _ref.min())


def hads_magnitude(phase, amplitude):
    """Asymmetric HADS light curve, in magnitudes relative to the mean.

    Returns a negative offset at maximum light (brighter = smaller magnitude),
    scaled so the peak-to-peak range is 2 * `amplitude`.
    """
    return -(amplitude / _TEMPLATE_SEMI) * _template(phase)


def _light_curve_mag(t_days, cfg: SynthConfig, t0: float):
    """Target magnitude at each epoch, including the second radial mode."""
    ph0 = ((t_days - t0) / cfg.period) % 1.0
    mag = cfg.mean_mag + hads_magnitude(ph0, cfg.amplitude)
    if cfg.second_mode:
        p1 = cfg.period * cfg.ratio_p1_p0
        ph1 = ((t_days - t0) / p1 + 0.31) % 1.0
        a1 = cfg.amplitude * cfg.amp_ratio_1
        mag = mag - a1 * np.sin(2 * np.pi * ph1)
    return mag


def _moffat(nx, ny, x0, y0, fwhm, beta=2.5, flux=1.0):
    """Moffat PSF stamp - broader wings than a Gaussian, like real seeing."""
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    yy, xx = np.mgrid[0:ny, 0:nx]
    r2 = (xx - x0) ** 2 + (yy - y0) ** 2
    prof = (1.0 + r2 / alpha ** 2) ** (-beta)
    norm = prof.sum()
    return prof * (flux / norm) if norm > 0 else prof


def _airmass(t_days, cfg: SynthConfig):
    """Rough airmass curve so the demo has a realistic slow trend to remove."""
    try:
        from astropy import units as u
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord

        loc = EarthLocation.from_geodetic(cfg.site["lon"] * u.deg,
                                          cfg.site["lat"] * u.deg,
                                          cfg.site["elev"] * u.m)
        coord = SkyCoord(TARGET_DEFAULT["ra_deg"] * u.deg,
                         TARGET_DEFAULT["dec_deg"] * u.deg, frame="icrs")
        times = Time(t_days, format="jd")
        alt = coord.transform_to(AltAz(obstime=times, location=loc)).alt.deg
        alt = np.clip(alt, 5.0, 90.0)
        return 1.0 / np.cos(np.radians(90.0 - alt)), alt
    except Exception:
        n = len(np.atleast_1d(t_days))
        return np.full(n, 1.25), np.full(n, 53.0)


def generate(outdir: str, cfg: SynthConfig = None, progress=None) -> dict:
    """Write a synthetic run to `outdir` and return a manifest."""
    cfg = cfg or SynthConfig()
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    size = int(cfg.size)
    n_total = int(cfg.n_frames)
    per_night = max(1, n_total // max(1, cfg.n_nights))

    t_start = Time(cfg.start_utc, scale="utc").jd
    times = []
    for night in range(cfg.n_nights):
        base = t_start + night * cfg.night_gap_days
        k = per_night if night < cfg.n_nights - 1 else n_total - per_night * (cfg.n_nights - 1)
        for i in range(k):
            times.append(base + (i * cfg.cadence + cfg.exptime / 2.0) / 86400.0)
    times = np.asarray(times, dtype=float)
    n_total = len(times)

    # ---- build the star field -------------------------------------------
    cx, cy = size / 2.0, size / 2.0
    margin = 55
    stars = [{"x": cx, "y": cy, "mag": cfg.mean_mag, "target": True, "name": "V0756 CrA"}]

    # A few deliberately bright, well-separated comparison stars.
    comp_mags = [11.20, 11.45, 11.80, 12.20, 12.50]
    angles = rng.permutation(np.linspace(0, 2 * np.pi, len(comp_mags), endpoint=False))
    for i, m in enumerate(comp_mags):
        r = rng.uniform(0.22, 0.40) * size
        a = angles[i] + rng.uniform(-0.25, 0.25)
        stars.append({
            "x": float(np.clip(cx + r * np.cos(a), margin, size - margin)),
            "y": float(np.clip(cy + r * np.sin(a), margin, size - margin)),
            "mag": float(m), "target": False, "name": f"comp{i + 1}",
        })

    n_extra = max(0, cfg.n_field_stars - len(comp_mags))
    for i in range(n_extra):
        for _ in range(60):
            x = rng.uniform(margin, size - margin)
            y = rng.uniform(margin, size - margin)
            if all((x - s["x"]) ** 2 + (y - s["y"]) ** 2 > 24 ** 2 for s in stars):
                break
        stars.append({"x": float(x), "y": float(y),
                      "mag": float(rng.uniform(12.6, 15.3)),
                      "target": False, "name": f"field{i + 1}"})

    # ---- per-frame conditions -------------------------------------------
    t0 = times[0]
    hours = (times - t0) * 24.0
    target_mag = _light_curve_mag(times, cfg, t0)
    airmass, altitude = _airmass(times, cfg)

    seeing = cfg.fwhm + 0.5 * cfg.seeing_variation * np.sin(
        2 * np.pi * hours / 1.7 + 0.6) + rng.normal(0, 0.06, n_total)
    seeing = np.clip(seeing, 2.0, 12.0)

    transp = 1.0 + rng.normal(0, cfg.transparency_rms, n_total)
    transp += 0.012 * np.sin(2 * np.pi * hours / 2.9)
    # Extinction with airmass: k_V ~ 0.15 mag/airmass at a good site.
    transp *= 10 ** (-0.4 * 0.15 * (airmass - airmass.min()))
    for _ in range(int(cfg.cloud_events)):
        c = rng.integers(int(0.15 * n_total), int(0.9 * n_total))
        width = max(2, int(0.02 * n_total))
        idx = np.arange(n_total)
        transp *= 1.0 - 0.32 * np.exp(-0.5 * ((idx - c) / width) ** 2)
    transp = np.clip(transp, 0.05, None)

    # Drift resets each night (the telescope re-points).
    night_index = np.zeros(n_total, dtype=int)
    if cfg.n_nights > 1:
        gaps = np.diff(times)
        night_index = np.concatenate([[0], np.cumsum(gaps > 0.25)])
    dx = np.zeros(n_total)
    dy = np.zeros(n_total)
    for nb in np.unique(night_index):
        sel = night_index == nb
        h = hours[sel] - hours[sel][0]
        dx[sel] = cfg.drift_x * h + rng.uniform(-6, 6)
        dy[sel] = cfg.drift_y * h + rng.uniform(-6, 6)
    dx += rng.normal(0, cfg.jitter, n_total) + 1.4 * np.sin(2 * np.pi * hours / 0.55)
    dy += rng.normal(0, cfg.jitter, n_total) + 1.1 * np.cos(2 * np.pi * hours / 0.61)

    # ---- render ---------------------------------------------------------
    yy = np.arange(size)
    grad = 1.0 + cfg.sky_gradient * (yy / size - 0.5)[:, None] * np.ones((1, size))
    grad = grad * (1.0 + cfg.sky_gradient * 0.5 * (np.arange(size) / size - 0.5)[None, :])

    files = []
    truth_rows = []
    stamp = 41
    half = stamp // 2

    for i in range(n_total):
        if progress:
            progress(i, n_total, f"frame {i + 1}/{n_total}")
        img = cfg.sky * grad * (transp[i] ** 0.35)

        for si, s in enumerate(stars):
            mag = target_mag[i] if s["target"] else s["mag"]
            flux = 10 ** (0.4 * (cfg.zeropoint - mag)) * cfg.exptime * transp[i]
            sx = s["x"] + dx[i]
            sy = s["y"] + dy[i]
            xi, yi = int(round(sx)), int(round(sy))
            x0, x1 = xi - half, xi + half + 1
            y0, y1 = yi - half, yi + half + 1
            if x1 <= 0 or y1 <= 0 or x0 >= size or y0 >= size:
                continue
            ps = _moffat(stamp, stamp, half + (sx - xi), half + (sy - yi),
                         seeing[i], flux=flux)
            gx0, gx1 = max(0, x0), min(size, x1)
            gy0, gy1 = max(0, y0), min(size, y1)
            img[gy0:gy1, gx0:gx1] += ps[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]

        if cfg.noise:
            electrons = np.clip(img, 0, None) * cfg.gain
            img = rng.poisson(electrons).astype(np.float64) / cfg.gain
            img += rng.normal(0, cfg.read_noise / cfg.gain, img.shape)
            # A handful of hot pixels, as any real sensor has.
            if i == 0:
                hot_y = rng.integers(0, size, 25)
                hot_x = rng.integers(0, size, 25)
            img[hot_y, hot_x] += rng.uniform(2000, 9000, len(hot_x))

        img = np.clip(img, 0, cfg.saturation)
        data = img.astype(np.uint16)

        hdr = fits.Header()
        t = Time(times[i] - cfg.exptime / 2.0 / 86400.0, format="jd", scale="utc")
        hdr["SIMPLE"] = True
        hdr["OBJECT"] = TARGET_DEFAULT["name"]
        hdr["DATE-OBS"] = t.isot
        hdr["EXPTIME"] = (cfg.exptime, "seconds")
        hdr["FILTER"] = "V"
        hdr["IMAGETYP"] = "LIGHT"
        hdr["GAIN"] = (cfg.gain, "e-/ADU")
        hdr["RDNOISE"] = (cfg.read_noise, "e-")
        hdr["SATURATE"] = cfg.saturation
        hdr["OBJCTRA"] = TARGET_DEFAULT["ra_str"]
        hdr["OBJCTDEC"] = TARGET_DEFAULT["dec_str"]
        hdr["SITELAT"] = cfg.site["lat"]
        hdr["SITELONG"] = cfg.site["lon"]
        hdr["SITEELEV"] = cfg.site["elev"]
        hdr["AIRMASS"] = round(float(airmass[i]), 4)
        hdr["TELESCOP"] = cfg.site["name"]
        hdr["INSTRUME"] = "VarStar Lab simulator"
        hdr["FWHM"] = round(float(seeing[i]), 3)
        hdr["COMMENT"] = "SYNTHETIC DATA - generated for pipeline testing"
        hdr["SIMPER"] = (cfg.period, "injected period, days")
        hdr["SIMAMP"] = (cfg.amplitude, "injected semi-amplitude, mag")
        hdr["SIMMAG"] = (round(float(target_mag[i]), 5), "true target mag this frame")

        name = f"v0756cra_{i + 1:04d}.fits"
        path = os.path.join(outdir, name)
        fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)
        files.append(path)
        truth_rows.append({
            "frame": i + 1, "jd": float(times[i]),
            "true_mag": float(target_mag[i]),
            "airmass": float(airmass[i]), "altitude_deg": float(altitude[i]),
            "fwhm_px": float(seeing[i]), "transparency": float(transp[i]),
            "dx": float(dx[i]), "dy": float(dy[i]),
        })

    span = float(times[-1] - times[0])
    return {
        "files": files,
        "n_frames": n_total,
        "outdir": outdir,
        "truth": {
            "period": cfg.period,
            "period_overtone": cfg.period * cfg.ratio_p1_p0 if cfg.second_mode else None,
            "amplitude_semi": cfg.amplitude,
            "amplitude_p2p": 2 * cfg.amplitude,
            "mean_mag": cfg.mean_mag,
            "comp_mags": {f"comp{i + 1}": m for i, m in enumerate(comp_mags)},
            "target_xy": [cx, cy],
            # Nominal positions, plus where everything actually sat on frame 1
            # once the un-guided pointing offset is applied.
            "stars": [dict(s) for s in stars],
            "stars_frame1": [
                {**s, "x": s["x"] + float(dx[0]), "y": s["y"] + float(dy[0])}
                for s in stars
            ],
            "frame1_offset": [float(dx[0]), float(dy[0])],
            "second_mode": cfg.second_mode,
            "rows": truth_rows,
        },
        "span_days": span,
        "span_hours": span * 24.0,
        "cycles": span / cfg.period,
        "site": cfg.site,
        "start_utc": cfg.start_utc,
    }


PRESETS = {
    "single_night": dict(
        n_frames=200, cadence=45.0, exptime=40.0, n_nights=1,
        label="One night, 2.5 h - what the proposal currently plans",
        description=("200 frames at 45 s cadence over 2.5 h: about 0.97 of a "
                     "pulsation cycle. Recovers the period to a few percent, and "
                     "shows why a single short run cannot do better."),
    ),
    "short_night": dict(
        n_frames=100, cadence=45.0, exptime=40.0, n_nights=1,
        label="One short night, 1.25 h - deliberately inadequate",
        description=("Half a pulsation cycle. The periodogram peak is set by the "
                     "length of the run rather than by the star - worth seeing once "
                     "so you recognise it."),
    ),
    "alias_trap": dict(
        n_frames=324, cadence=50.0, exptime=40.0, n_nights=3, night_gap_days=1.0,
        label="Three SHORT nights - the one-day alias trap",
        description=("Three 1.5 h sessions on consecutive nights. 19 cycles of "
                     "baseline, hundreds of points, a false-alarm probability of "
                     "1e-180 - and the period is wrong by 9%, because the fit locks "
                     "onto the +1 cycle/day alias. Adding more short nights makes "
                     "this worse, not better."),
    ),
    "three_nights": dict(
        n_frames=648, cadence=50.0, exptime=40.0, n_nights=3, night_gap_days=1.0,
        label="Three nights x 3 h - a period you can actually quote",
        description=("Three 3 h sessions, each covering 1.2 cycles. Covering a full "
                     "cycle inside one continuous session is what kills the 1 "
                     "cycle/day alias; the 2-day baseline then supplies precision."),
    ),
    "clean": dict(
        n_frames=430, cadence=40.0, exptime=35.0, n_nights=1,
        transparency_rms=0.004, cloud_events=0, drift_x=2.0, drift_y=1.0,
        jitter=0.1, read_noise=4.0, seeing_variation=0.15,
        label="Idealised single night, 4.8 h - the pipeline floor",
        description=("Near-perfect conditions, auto-guiding on, 1.9 cycles in one "
                     "continuous run. Whatever error you get here is the noise floor "
                     "of the reduction itself."),
    ),
}


def config_from_preset(name: str, **overrides) -> SynthConfig:
    preset = dict(PRESETS.get(name, PRESETS["single_night"]))
    preset.pop("label", None)
    preset.pop("description", None)
    preset.update({k: v for k, v in overrides.items() if v is not None})
    valid = {f for f in SynthConfig.__dataclass_fields__}
    return SynthConfig(**{k: v for k, v in preset.items() if k in valid})
