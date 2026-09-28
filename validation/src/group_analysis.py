"""Subject-level Welch tests; one BH family per dataset including global delay."""

import numpy as np


def bh_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or not len(pvalues):
        raise ValueError("BH needs a nonempty 1-D family")
    finite = np.isfinite(pvalues)
    if ((pvalues[finite] < 0) | (pvalues[finite] > 1)).any():
        raise ValueError("Invalid p value")
    # Undefined tests stay in the predeclared family as p=1 for correction.
    work = np.where(finite, pvalues, 1.0)
    order = np.argsort(work, kind="stable")
    ranked = work[order] * len(work) / np.arange(1, len(work) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    result[~finite] = np.nan
    return result


def subject_parameters(data, dataset):
    rows, arrays, labels = [], [], []
    for subject in np.unique(data["subject_id"][data["dataset"] == dataset]):
        take = (data["dataset"] == dataset) & (data["subject_id"] == subject)
        label = np.unique(data["label"][take])
        if len(label) != 1:
            raise ValueError(f"Inconsistent label: {subject}")
        rows.append(
            {
                "dataset": dataset,
                "subject_id": str(subject),
                "label": int(label[0]),
                "n_segments": int(take.sum()),
            }
        )
        arrays.append(data["features"][take].mean(axis=0, dtype=np.float64))
        labels.append(label[0])
    return np.stack(arrays), np.asarray(labels), rows


def welch_statistics(features, labels):
    from scipy.stats import t as student_t

    case, control = features[labels == 1], features[labels == 0]
    n1, n0 = len(case), len(control)
    if min(n1, n0) < 2:
        raise ValueError("Welch tests need at least two subjects per group")
    m1, m0 = case.mean(0), control.mean(0)
    v1, v0 = case.var(0, ddof=1), control.var(0, ddof=1)
    se2 = v1 / n1 + v0 / n0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (m1 - m0) / np.sqrt(se2)
        df = se2**2 / ((v1 / n1) ** 2 / (n1 - 1) + (v0 / n0) ** 2 / (n0 - 1))
        pooled = ((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2)
        effect = (m1 - m0) / np.sqrt(pooled)
    valid = (se2 > 0) & np.isfinite(df)
    t, df = np.where(valid, t, np.nan), np.where(valid, df, np.nan)
    p = 2 * student_t.sf(np.abs(t), df)
    return {
        "case_mean": m1,
        "control_mean": m0,
        "t": t,
        "dof": df,
        "p": p,
        "q": bh_adjust(p),
        "cohen_d": np.where(pooled > 0, effect, np.nan),
        "n_case": np.full(len(m1), n1),
        "n_control": np.full(len(m1), n0),
    }
