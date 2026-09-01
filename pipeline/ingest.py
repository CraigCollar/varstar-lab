"""Frame ingestion: read FITS / camera images, recover mid-exposure timestamps.

Supports the observing setup described in the project proposal (Prasarttongosoth,
Hawkins & Collar): ~30-45 s unstacked exposures, 0 s delay, taken continuously for
2-3 h with auto-guiding OFF (so frames drift and must be re-registered downstream).
"""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.time import Time

warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)

FITS_EXT = {".fit", ".fits", ".fts", ".fit.gz", ".fits.gz"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
RAW_EXT = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".raf", ".rw2", ".pef"}

# Header keys that may carry an absolute time, best first.
_JD_KEYS = ("BJD_TDB", "BJD", "HJD", "JD-OBS", "JD_OBS", "JD", "JULIAN")
_MJD_KEYS = ("MJD-OBS", "MJD_OBS", "MJD-MID", "MJD")
_DATE_KEYS = ("DATE-OBS", "DATE_OBS", "DATE-BEG", "DATEOBS", "UTSTART", "DATE-AVG")
_EXP_KEYS = ("EXPTIME", "EXPOSURE", "EXP_TIME", "ITIME", "TELAPSE", "XPOSURE")
_FILTER_KEYS = ("FILTER", "FILTER1", "FILTNAM", "INSFLNAM")
_GAIN_KEYS = ("GAIN", "EGAIN", "GAINRAW")
_CCDTEMP_KEYS = ("CCD-TEMP", "CCDTEMP", "SET-TEMP")
_OBJECT_KEYS = ("OBJECT", "OBJCTNAM", "TARGET")


@dataclass
class Frame:
    """One science exposure and everything we know about when it was taken."""

    index: int
    filename: str
    path: str
    shape: tuple = (0, 0)
    kind: str = "fits"            # fits | image | raw
    jd: Optional[float] = None    # mid-exposure, UTC-based Julian Date
    time_source: str = "none"     # where jd came from
    exptime: Optional[float] = None
    filter_name: Optional[str] = None
    gain_hdr: Optional[float] = None
    ccd_temp: Optional[float] = None
    object_name: Optional[str] = None
    channel_used: Optional[str] = None
    is_color: bool = False
    bitdepth: Optional[int] = None
    saturation: Optional[float] = None
    note: str = ""
    excluded: bool = False
    header_preview: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d


class IngestError(Exception):
    pass


# --------------------------------------------------------------------------
# pixel data
# --------------------------------------------------------------------------

def _pick_channel(cube: np.ndarray, channel: str) -> tuple[np.ndarray, str]:
    """Collapse an (H, W, 3|4) colour array to 2-D.

    The green channel is the closest single-channel match to Johnson V for a
    typical Bayer-filtered CMOS/DSLR sensor, so it is the default.
    """
    channel = (channel or "G").upper()
    if cube.ndim == 2:
        return cube, "mono"
    if cube.ndim != 3:
        raise IngestError(f"cannot handle array with shape {cube.shape}")

    # Some FITS cubes are (3, H, W) rather than (H, W, 3).
    if cube.shape[0] in (3, 4) and cube.shape[-1] not in (3, 4):
        cube = np.moveaxis(cube, 0, -1)

    n = cube.shape[-1]
    if n < 3:
        return cube[..., 0], "plane0"

    if channel == "R":
        return cube[..., 0], "R"
    if channel == "B":
        return cube[..., 2], "B"
    if channel in ("L", "LUM", "LUMINANCE"):
        # Rec.709 luma
        lum = (0.2126 * cube[..., 0] + 0.7152 * cube[..., 1] + 0.0722 * cube[..., 2])
        return lum, "luminance"
    if channel in ("SUM", "CLEAR"):
        return cube[..., :3].sum(axis=-1), "R+G+B"
    return cube[..., 1], "G"


def _read_fits(path: str, channel: str):
    with fits.open(path, memmap=False) as hdul:
        hdu = None
        for h in hdul:
            if getattr(h, "data", None) is not None and np.asarray(h.data).ndim >= 2:
                hdu = h
                break
        if hdu is None:
            raise IngestError("no image data found in FITS file")
        data = np.asarray(hdu.data, dtype=np.float64)
        header = dict(hdu.header)
        # Merge primary header keywords if we used an extension.
        if hdu is not hdul[0]:
            prim = dict(hdul[0].header)
            for k, v in prim.items():
                header.setdefault(k, v)

    is_color = data.ndim == 3 and (3 in data.shape or 4 in data.shape)
    if data.ndim > 2:
        if is_color:
            data, used = _pick_channel(data, channel)
        else:
            # A time cube: use the first plane and warn upstream.
            data, used = data.reshape(-1, *data.shape[-2:])[0], "cube-plane0"
    else:
        used = "mono"

    bzero = header.get("BZERO", 0.0)
    bitpix = header.get("BITPIX")
    sat = None
    if isinstance(bitpix, int) and bitpix > 0:
        sat = float(2 ** min(bitpix, 32) - 1)
        if bzero:  # unsigned data stored as signed
            sat = float(2 ** min(bitpix, 32) - 1)
    for key in ("SATURATE", "DATAMAX", "MAXLIN"):
        if key in header:
            try:
                sat = float(header[key])
                break
            except (TypeError, ValueError):
                pass
    return data, header, is_color, used, sat


def _read_image(path: str, channel: str):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        exif = _pil_exif(im)
        mode = im.mode
        if mode in ("I;16", "I;16B", "I;16L", "I", "F", "L"):
            arr = np.asarray(im, dtype=np.float64)
            is_color = False
        else:
            if mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float64)
            is_color = arr.ndim == 3

    maxval = float(np.nanmax(arr)) if arr.size else 0.0
    bitdepth = 16 if maxval > 255 else 8
    sat = float(2 ** bitdepth - 1)

    data, used = _pick_channel(arr, channel) if is_color else (arr, "mono")
    header = {f"EXIF_{k}": v for k, v in exif.items()}
    header["_BITDEPTH"] = bitdepth
    return data, header, is_color, used, sat


def _read_raw(path: str, channel: str):
    try:
        import rawpy
    except ImportError:
        raise IngestError(
            "camera RAW files need the 'rawpy' package (pip install rawpy). "
            "Export to 16-bit TIFF or FITS instead - and prefer FITS, which keeps "
            "the linear sensor values photometry requires."
        )
    with rawpy.imread(path) as raw:
        # Linear, no white balance / gamma / auto-brightness: photometry needs
        # the sensor response to stay proportional to photon count.
        rgb = raw.postprocess(
            gamma=(1, 1), no_auto_bright=True, output_bps=16,
            use_camera_wb=False, use_auto_wb=False, no_auto_scale=False,
        )
        white = float(raw.white_level or 65535)
    arr = np.asarray(rgb, dtype=np.float64)
    data, used = _pick_channel(arr, channel)
    header = {f"EXIF_{k}": v for k, v in _exif_from_file(path).items()}
    return data, header, True, used, white


def _pil_exif(im) -> dict:
    out = {}
    try:
        from PIL import ExifTags

        raw = im.getexif()
        if not raw:
            return out
        for tag, val in raw.items():
            out[ExifTags.TAGS.get(tag, str(tag))] = val
        for ifd_name, ifd_id in (("Exif", 0x8769), ("GPS", 0x8825)):
            try:
                ifd = raw.get_ifd(ifd_id)
            except Exception:
                continue
            table = ExifTags.GPSTAGS if ifd_name == "GPS" else ExifTags.TAGS
            for tag, val in (ifd or {}).items():
                out[table.get(tag, str(tag))] = val
    except Exception:
        pass
    return out


def _exif_from_file(path: str) -> dict:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return _pil_exif(im)
    except Exception:
        return {}


# --------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------

def _as_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float, np.floating, np.integer)):
        f = float(v)
        return f if np.isfinite(f) else None
    try:
        # EXIF rationals
        if hasattr(v, "numerator") and hasattr(v, "denominator"):
            return float(v.numerator) / float(v.denominator or 1)
    except Exception:
        pass
    s = str(v).strip()
    m = re.match(r"^([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    if "/" in s:  # "1/125"
        try:
            a, b = s.split("/", 1)
            return float(a) / float(b)
        except Exception:
            return None
    return None


def _exptime_from(header: dict) -> Optional[float]:
    for k in _EXP_KEYS:
        if k in header:
            v = _as_float(header[k])
            if v and v > 0:
                return v
    for k in ("EXIF_ExposureTime", "EXIF_ShutterSpeedValue"):
        if k in header:
            v = _as_float(header[k])
            if v and v > 0:
                return v
    return None


def _parse_exif_datetime(s: str, subsec=None) -> Optional[float]:
    s = str(s).strip()
    m = re.match(r"^(\d{4})[:\-](\d{2})[:\-](\d{2})[ T](\d{2}):(\d{2}):(\d{2})", s)
    if not m:
        return None
    y, mo, d, h, mi, sec = (int(x) for x in m.groups())
    frac = 0.0
    if subsec is not None:
        try:
            frac = float(f"0.{str(subsec).strip()}")
        except ValueError:
            frac = 0.0
    try:
        dt = datetime(y, mo, d, h, mi, sec, tzinfo=timezone.utc)
    except ValueError:
        return None
    return Time(dt, scale="utc").jd + frac / 86400.0


def extract_time(header: dict) -> tuple[Optional[float], str, Optional[float]]:
    """Return (mid-exposure JD, description of where it came from, exptime).

    A start-of-exposure timestamp is shifted by +exptime/2 so every frame is
    tagged with the flux-weighted mean epoch of its own integration. Skipping
    this shift biases every point by a constant, which is harmless for period
    finding but wrong for a time of maximum.
    """
    exptime = _exptime_from(header)

    for k in _JD_KEYS:
        if k in header:
            v = _as_float(header[k])
            if v and v > 2_000_000:
                mid = "" if k in ("DATE-AVG",) else ""
                return v, f"header {k} (assumed mid-exposure)" + mid, exptime
            if v and 0 < v < 100_000:  # reduced JD
                return v + 2_400_000.5, f"header {k} (+2400000.5)", exptime

    for k in _MJD_KEYS:
        if k in header:
            v = _as_float(header[k])
            if v and 10_000 < v < 100_000:
                jd = v + 2_400_000.5
                if k in ("MJD-MID",):
                    return jd, f"header {k}", exptime
                if exptime:
                    return jd + exptime / 2.0 / 86400.0, f"header {k} + exp/2", exptime
                return jd, f"header {k}", exptime

    for k in _DATE_KEYS:
        if k in header:
            try:
                t = Time(str(header[k]).strip(), format="isot", scale="utc")
            except Exception:
                try:
                    t = Time(str(header[k]).strip(), scale="utc")
                except Exception:
                    continue
            jd = float(t.jd)
            if k == "DATE-AVG":
                return jd, f"header {k}", exptime
            if exptime:
                return jd + exptime / 2.0 / 86400.0, f"header {k} + exp/2", exptime
            return jd, f"header {k} (no EXPTIME; start of exposure)", exptime

    sub = header.get("EXIF_SubsecTimeOriginal") or header.get("EXIF_SubsecTime")
    for k in ("EXIF_DateTimeOriginal", "EXIF_DateTimeDigitized", "EXIF_DateTime"):
        if k in header:
            jd = _parse_exif_datetime(header[k], sub)
            if jd:
                label = f"EXIF {k.replace('EXIF_', '')} (assumed UTC)"
                if exptime:
                    return jd + exptime / 2.0 / 86400.0, label + " + exp/2", exptime
                return jd, label, exptime
    return None, "none", exptime


def times_from_filenames(frames: list[Frame]) -> bool:
    """Sort frames by a trailing sequence number in the filename, if present."""
    pat = re.compile(r"(\d{2,})(?=\D*$)")
    keys = []
    for f in frames:
        m = pat.search(os.path.splitext(f.filename)[0])
        if not m:
            return False
        keys.append(int(m.group(1)))
    if len(set(keys)) != len(keys):
        return False
    order = np.argsort(keys)
    for new_i, old_i in enumerate(order):
        frames[old_i].index = int(new_i)
    frames.sort(key=lambda fr: fr.index)
    return True


def apply_manual_times(frames: list[Frame], start_utc: str, cadence_s: float,
                       exptime_s: Optional[float] = None) -> None:
    """Stamp frames with a uniform cadence starting at `start_utc`.

    Used when the images carry no usable timestamp (stripped EXIF, screenshots,
    etc.). Times are mid-exposure: start + exp/2 + i*cadence.
    """
    t0 = Time(str(start_utc).strip().replace(" ", "T"), scale="utc").jd
    half = (exptime_s or 0.0) / 2.0 / 86400.0
    for i, f in enumerate(frames):
        f.jd = t0 + half + i * float(cadence_s) / 86400.0
        f.time_source = f"manual: {start_utc} UTC + {cadence_s:g}s cadence"
        if exptime_s:
            f.exptime = float(exptime_s)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def classify(path: str) -> str:
    low = path.lower()
    if low.endswith(".gz"):
        low = low[:-3]
    ext = os.path.splitext(low)[1]
    if ext in FITS_EXT or low.endswith((".fit", ".fits", ".fts")):
        return "fits"
    if ext in RAW_EXT:
        return "raw"
    if ext in IMAGE_EXT:
        return "image"
    return "unknown"


def read_pixels(path: str, channel: str = "G"):
    """Return (2-D float array, header dict, is_color, channel_used, saturation)."""
    kind = classify(path)
    if kind == "fits":
        return _read_fits(path, channel)
    if kind == "image":
        return _read_image(path, channel)
    if kind == "raw":
        return _read_raw(path, channel)
    raise IngestError(f"unsupported file type: {os.path.basename(path)}")


_PREVIEW_KEYS = (
    "OBJECT", "DATE-OBS", "EXPTIME", "FILTER", "GAIN", "EGAIN", "CCD-TEMP",
    "INSTRUME", "TELESCOP", "IMAGETYP", "XBINNING", "YBINNING", "FOCALLEN",
    "OBSGEO-B", "OBSGEO-L", "SITELAT", "SITELONG", "SITEELEV", "AIRMASS",
    "OBJCTRA", "OBJCTDEC", "RA", "DEC",
    "EXIF_Model", "EXIF_Make", "EXIF_DateTimeOriginal", "EXIF_ExposureTime",
    "EXIF_ISOSpeedRatings", "EXIF_FocalLength",
)


def scan_frame(path: str, index: int, channel: str = "G") -> Frame:
    """Read one file's metadata (and shape) without keeping the pixels."""
    fname = os.path.basename(path)
    kind = classify(path)
    fr = Frame(index=index, filename=fname, path=path, kind=kind)
    try:
        data, header, is_color, used, sat = read_pixels(path, channel)
    except Exception as exc:
        fr.note = f"unreadable: {exc}"
        fr.excluded = True
        return fr

    fr.shape = tuple(int(x) for x in data.shape)
    fr.is_color = bool(is_color)
    fr.channel_used = used
    fr.saturation = float(sat) if sat else None
    fr.bitdepth = header.get("_BITDEPTH")

    jd, src, exptime = extract_time(header)
    fr.jd, fr.time_source, fr.exptime = jd, src, exptime

    for k in _FILTER_KEYS:
        if header.get(k):
            fr.filter_name = str(header[k]).strip()
            break
    for k in _GAIN_KEYS:
        if k in header:
            fr.gain_hdr = _as_float(header[k])
            break
    for k in _CCDTEMP_KEYS:
        if k in header:
            fr.ccd_temp = _as_float(header[k])
            break
    for k in _OBJECT_KEYS:
        if header.get(k):
            fr.object_name = str(header[k]).strip()
            break

    fr.header_preview = {
        k: (str(v)[:80] if not isinstance(v, (int, float)) else v)
        for k, v in header.items() if k in _PREVIEW_KEYS
    }
    return fr


def observatory_from_header(header: dict) -> Optional[dict]:
    """Pull an observing site out of FITS keywords, if it is there."""
    lat = lon = elev = None
    for k in ("SITELAT", "OBSGEO-B", "LAT-OBS", "LATITUDE", "OBSLAT"):
        if k in header:
            lat = _sexa(header[k])
            if lat is not None:
                break
    for k in ("SITELONG", "OBSGEO-L", "LONG-OBS", "LONGITUD", "OBSLONG"):
        if k in header:
            lon = _sexa(header[k])
            if lon is not None:
                break
    for k in ("SITEELEV", "OBSGEO-H", "ALT-OBS", "ELEVATIO", "HEIGHT"):
        if k in header:
            elev = _as_float(header[k])
            if elev is not None:
                break
    if lat is None or lon is None:
        return None
    return {"lat": lat, "lon": lon, "elev": elev if elev is not None else 0.0}


def _sexa(v) -> Optional[float]:
    """Parse a coordinate that may be decimal degrees or sexagesimal."""
    if v is None:
        return None
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    s = str(v).strip().replace("'", " ").replace('"', " ")
    s = re.sub(r"[dhms:]", " ", s, flags=re.I)
    parts = [p for p in s.split() if p]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return _as_float(v)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    sign = -1.0 if str(v).strip().startswith("-") else 1.0
    mag = abs(vals[0]) + (vals[1] / 60.0 if len(vals) > 1 else 0) + \
          (vals[2] / 3600.0 if len(vals) > 2 else 0)
    return sign * mag
