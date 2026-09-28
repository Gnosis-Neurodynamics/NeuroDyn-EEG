"""Unfiltered regression metrics and real-EEG inverse/forward metrics."""

import numpy as np


class RegressionMoments:
    """Mergeable centered moments: identical metrics across arbitrary chunk sizes."""

    def __init__(self):
        self.n = 0

    def update(self, truth, predicted):
        x, y = np.asarray(truth, dtype=float), np.asarray(predicted, dtype=float)
        if (
            x.shape != y.shape
            or x.ndim != 2
            or not len(x)
            or not (np.isfinite(x).all() and np.isfinite(y).all())
        ):
            raise ValueError(
                "Regression inputs require matching finite nonempty (N,D) arrays"
            )
        n = len(x)
        mx, my = x.mean(0), y.mean(0)
        vx, vy = ((x - mx) ** 2).sum(0), ((y - my) ** 2).sum(0)
        cov = ((x - mx) * (y - my)).sum(0)
        ae, se = np.abs(x - y).sum(0), ((x - y) ** 2).sum(0)
        if not self.n:
            self.mx, self.my, self.vx, self.vy, self.cov = mx, my, vx, vy, cov
            self.ae, self.se, self.n = ae, se, n
            return
        dx, dy = mx - self.mx, my - self.my
        weight = self.n * n / (self.n + n)
        self.vx += vx + dx * dx * weight
        self.vy += vy + dy * dy * weight
        self.cov += cov + dx * dy * weight
        self.mx += dx * n / (self.n + n)
        self.my += dy * n / (self.n + n)
        self.ae += ae
        self.se += se
        self.n += n

    def result(self):
        if not self.n:
            raise ValueError("No included samples")
        with np.errstate(divide="ignore", invalid="ignore"):
            r = self.cov / np.sqrt(self.vx * self.vy)
            r2 = 1 - self.se / self.vx
        r = np.where((self.vx > 0) & (self.vy > 0), np.clip(r, -1, 1), np.nan)
        r2 = np.where(self.vx > 0, r2, np.nan)
        return {
            "n": np.full(self.mx.shape, self.n),
            "pearson_r": r,
            "r2": r2,
            "mae": self.ae / self.n,
            "rmse": np.sqrt(self.se / self.n),
        }


class ParameterMetrics:
    def __init__(self, names):
        self.values = {
            (key, scope): RegressionMoments()
            for key in names
            for scope in ("pooled", "sample_mean", "region")
        }

    def update(self, truth, predicted):
        for (key, scope), stats in self.values.items():
            x, y = truth[key], predicted[key]
            if scope == "pooled":
                x, y = x.reshape(-1, 1), y.reshape(-1, 1)
            elif scope == "sample_mean":
                x, y = x.mean(1, keepdims=True), y.mean(1, keepdims=True)
            stats.update(x, y)

    def rows(self, **metadata):
        for (key, scope), stats in self.values.items():
            result = stats.result()
            for i in range(len(result["n"])):
                yield {
                    **metadata,
                    "parameter": key,
                    "scope": scope,
                    "region_index": i if scope == "region" else "",
                    **{name: value[i].item() for name, value in result.items()},
                }


def closed_loop_metrics(observed, reconstructed, fs=256):
    from scipy.signal import welch, butter, sosfiltfilt, hilbert

    if observed.shape != reconstructed.shape or observed.ndim != 3:
        raise ValueError(
            "Closed-loop shapes must match exactly; no time/sample cropping is allowed"
        )
    if not (np.isfinite(observed).all() and np.isfinite(reconstructed).all()):
        raise ValueError("Nonfinite closed-loop EEG")
    frequency, p = welch(observed, fs=fs, nperseg=2 * fs, noverlap=fs, axis=-1)
    _, q = welch(reconstructed, fs=fs, nperseg=2 * fs, noverlap=fs, axis=-1)
    selected = (frequency >= 1) & (frequency <= 40)
    logp, logq = np.log10(np.maximum(p[..., selected], 1e-30)), np.log10(
        np.maximum(q[..., selected], 1e-30)
    )
    xp, xq = logp - logp.mean(-1, keepdims=True), logq - logq.mean(-1, keepdims=True)
    denominator = np.sqrt((xp * xp).sum(-1) * (xq * xq).sum(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = (xp * xq).sum(-1) / denominator
    logf = np.log10(frequency[selected])
    logf -= logf.mean()
    slope_p, slope_q = (logp * logf).sum(-1) / (logf * logf).sum(), (logq * logf).sum(
        -1
    ) / (logf * logf).sum()
    alpha = (frequency >= 7) & (frequency <= 13)
    ap, aq = (
        frequency[alpha][p[..., alpha].argmax(-1)],
        frequency[alpha][q[..., alpha].argmax(-1)],
    )
    result = {
        "log_psd_r": correlation,
        "alpha_peak_abs_error_hz": np.abs(ap - aq),
        "slope_abs_error": np.abs(slope_p - slope_q),
    }
    for band, edges in {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13)}.items():
        sos = butter(4, edges, fs=fs, btype="bandpass", output="sos")
        phase_p = np.angle(hilbert(sosfiltfilt(sos, observed, axis=-1), axis=-1))
        phase_q = np.angle(hilbert(sosfiltfilt(sos, reconstructed, axis=-1), axis=-1))
        result[f"plv_{band}"] = np.abs(np.exp(1j * (phase_p - phase_q)).mean(-1))
    return result
