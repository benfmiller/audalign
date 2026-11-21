"""numpy versions of the audioop calls pydub makes when audalign reads a file.

Each function gives the same bytes as the audioop call it replaces (same integer and double
arithmetic), only vectorised. They handle 16-bit samples; callers fall back to pydub otherwise.
"""

import math

import numpy as np
from pydub.utils import db_to_float, ratio_to_db

_CHUNK = 1 << 20  # samples per chunk, bounds the float64 temporaries on long files


def _fbound(val, minval, maxval):
    # audioop.c fbound(): clamp, then floor
    val = np.where(val > maxval, maxval, np.where(val < minval + 1, minval, val))
    return np.floor(val)


def ratecv(frames, inrate, outrate):
    """audioop.ratecv(data, 2, channels, inrate, outrate, None) on int16 frames shaped (n, channels)."""
    g = math.gcd(inrate, outrate)
    inrate //= g
    outrate //= g
    n_in = frames.shape[0]
    n_out = ((n_in - 1) * outrate) // inrate + 1 if n_in else 0
    out = np.empty((n_out, frames.shape[1]), np.int16)
    for start in range(0, n_out, _CHUNK):
        # Output m interpolates inputs cur - 1 and cur with weight d / outrate; audioop
        # stops once it would need an input past the end.
        m = np.arange(start, min(start + _CHUNK, n_out), dtype=np.int64)
        cur_idx = -((-m * inrate) // outrate)
        d = (cur_idx * outrate - m * inrate).astype(np.float64)[:, None]
        cur = frames[cur_idx] * 65536.0  # samples are widened to 32 bits (<< 16)
        prev = frames[cur_idx - 1] * 65536.0
        prev[cur_idx == 0] = 0.0
        value = np.trunc((prev * d + cur * (outrate - d)) / outrate)
        out[start : start + len(m)] = value.astype(np.int64) >> 16
    return out


def tomono(frames, lfactor=0.5, rfactor=0.5):
    """audioop.tomono(data, 2, lfactor, rfactor) on int16 frames shaped (n, 2)."""
    out = np.empty(frames.shape[0], np.int16)
    for start in range(0, len(out), _CHUNK):
        chunk = frames[start : start + _CHUNK]
        out[start : start + len(chunk)] = _fbound(chunk[:, 0] * lfactor + chunk[:, 1] * rfactor, -0x8000, 0x7FFF)
    return out


def mul(samples, factor):
    """audioop.mul(data, 2, factor) on int16 samples."""
    out = np.empty_like(samples)
    for start in range(0, len(out), _CHUNK):
        chunk = samples[start : start + _CHUNK]
        out[start : start + len(chunk)] = _fbound(chunk * factor, -0x8000, 0x7FFF)
    return out


def normalize(samples, headroom=0.1):
    """pydub.effects.normalize(segment, headroom) on int16 samples."""
    peak = int(np.max(np.abs(samples.astype(np.int32)))) if len(samples) else 0  # audioop.max
    if peak == 0:
        return samples
    target_peak = 32768.0 * db_to_float(-headroom)  # max_possible_amplitude for 16 bits
    return mul(samples, db_to_float(float(ratio_to_db(target_peak / peak))))
