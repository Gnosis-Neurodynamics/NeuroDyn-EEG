"""Per-trial, exactly SNR-scaled surrogate noise following supplement S6."""

import numpy as np
from .common import seed_for


def band_noise(rng, shape, fs, low, high):
    freq = np.fft.rfftfreq(shape[-1], 1 / fs)
    spectrum = np.fft.rfft(rng.normal(size=shape), axis=-1)
    spectrum *= (freq >= low) & (freq <= high)
    return np.fft.irfft(spectrum, n=shape[-1], axis=-1)


def unit_noise(kind, shape, rng, options, fs=256):
    channels, length = shape
    if kind == "white":
        noise = rng.normal(size=shape)
    elif kind == "pink":
        freq = np.fft.rfftfreq(length, 1 / fs)
        scale = np.zeros_like(freq)
        scale[1:] = freq[1:] ** -0.5
        noise = np.fft.irfft(
            np.fft.rfft(rng.normal(size=shape), axis=-1) * scale, n=length, axis=-1
        )
    elif kind == "emg":
        width = max(3, int(fs * options["emg_envelope_width_seconds"]))
        if width > length:
            raise ValueError("EMG envelope is wider than a segment")
        impulses = rng.poisson(options["emg_envelope_rate_hz"] / fs, size=shape)
        envelope = np.stack(
            [np.convolve(row, np.hanning(width), mode="same") for row in impulses]
        )
        # Small tonic activity keeps a segment with zero bursts well defined.
        noise = band_noise(rng, shape, fs, 20, 100) * (0.05 + envelope)
    elif kind == "eog":
        noise = band_noise(rng, shape, fs, 0.01, 4)
        noise /= max(float(noise.std()), 1e-12)
        for _ in range(rng.poisson(options["eog_blink_rate_hz"] * length / fs)):
            width = min(length, max(3, int(rng.uniform(0.1, 0.4) * fs)))
            start = rng.integers(0, length - width + 1)
            noise[:, start : start + width] += (
                0.5 * rng.uniform(0.8, 1.2, (channels, 1)) * np.hanning(width)
            )
    else:
        raise ValueError(f"Unknown noise family: {kind}")
    return noise - noise.mean(axis=-1, keepdims=True)


def inject_noise(eeg, ids, family, snr_db, options):
    if len(eeg) != len(ids):
        raise ValueError("Noise needs one stable ID per trial")
    result = np.empty_like(eeg, dtype=np.float64)
    for i, (clean, trial_id) in enumerate(zip(eeg, ids)):
        # Same realization at different SNRs; independent of chunk/batch ordering.
        rng = np.random.default_rng(seed_for(options["seed"], family, str(trial_id)))
        noise = unit_noise(family, clean.shape, rng, options)
        signal_power, noise_power = np.mean(clean.astype(float) ** 2), np.mean(noise**2)
        if signal_power <= 0 or noise_power <= 0:
            raise ValueError(f"Zero power for trial {trial_id}")
        result[i] = clean + noise * np.sqrt(
            signal_power / (10 ** (snr_db / 10) * noise_power)
        )
    return result
