"""Parameter specifications for hierarchical physiology-aware sampling."""

from __future__ import annotations

from typing import Dict

# Physiologically motivated coupling constant from David & Friston 2003,
# also used by RetrainWorkers/src/model.py ConstrainedLayer (k = 12/5).
TAU_RATIO_K = 12.0 / 5.0
TAU_RATIO_MIN = 1.0  #

# ---------------------------------------------------------------------------
# Spec dict
# ---------------------------------------------------------------------------
# 注: bounds 与 Trainworker/src/config_params.py
PARAM_SPECS: Dict[str, dict] = {
    # ----- Time constants (slow rhythm, ~10 Hz alpha) -----
    "tau_e1": {
        "bounds": (1.0, 3.5),
        "shape": (90,),
        "distribution": "normal_truncated",
        "subject_std_frac": 1 / 4,
        "node_std_frac": 1 / 12,
        "long_name": "Excitatory time constant, slow rhythm (~10 Hz alpha)",
        "physical_unit": "code units (= ms / 10)",
    },
    "tau_i1": {
        "bounds": (1.0, 3.5),
        "shape": (90,),
        "distribution": "normal_truncated",
        "subject_std_frac": 1 / 4,
        "node_std_frac": 1 / 12,
        "coupled_with": "tau_e1",  # tau_i1 ∈ [low, k * tau_e1] ∩ bounds
        "long_name": "Inhibitory time constant, slow rhythm",
        "physical_unit": "code units (= ms / 10)",
    },
    # ----- Time constants (fast rhythm, ~40 Hz gamma) -----
    "tau_e2": {
        "bounds": (0.39, 0.84),
        "shape": (90,),
        "distribution": "normal_truncated",
        "subject_std_frac": 1 / 4,
        "node_std_frac": 1 / 12,
        "long_name": "Excitatory time constant, fast rhythm (~40 Hz gamma)",
        "physical_unit": "code units (= ms / 10)",
    },
    "tau_i2": {
        "bounds": (0.73, 1.68),
        "shape": (90,),
        "distribution": "normal_truncated",
        "subject_std_frac": 1 / 4,
        "node_std_frac": 1 / 12,
        "coupled_with": "tau_e2",
        "long_name": "Inhibitory time constant, fast rhythm",
        "physical_unit": "code units (= ms / 10)",
    },
    # ----- Sigmoid threshold / slope / max-rate -----
    "theta": {
        "bounds": (4.0, 7.0),
        "shape": (90,),
        "distribution": "normal_truncated",
        "subject_std_frac": 1 / 4,
        "node_std_frac": 1 / 8,
        "long_name": "Firing threshold v0 of sigmoid",
        "physical_unit": "mV",
    },
    "beta": {
        "bounds": (0.4, 0.7),
        "shape": (90,),
        "distribution": "normal_truncated",
        "subject_std_frac": 1 / 4,
        "node_std_frac": 1 / 8,
        "long_name": "Sigmoid slope r",
        "physical_unit": "1/mV",
    },
    "rmax": {
        "bounds": (4.0, 7.0),
        "shape": (90,),
        "distribution": "normal_truncated",
        "subject_std_frac": 1 / 4,
        "node_std_frac": 1 / 8,
        "long_name": "Maximum firing rate e0",
        "physical_unit": "Hz",
    },
    # ----- Synaptic connectivity -----
    "C1": {
        "bounds": (0.5, 1.5),
        "shape": (90,),
        "distribution": "normal_truncated",
        "subject_std_frac": 1 / 4,
        "node_std_frac": 1 / 6,
        "long_name": "Pyramidal -> Stellate",
        "physical_unit": "× 135 synapses (handled inside JR)",
    },
    "C2": {
        "bounds": (0.4, 1.2),
        "shape": (90,),
        "distribution": "normal_truncated",
        "subject_std_frac": 1 / 4,
        "node_std_frac": 1 / 6,
        "long_name": "Stellate -> Pyramidal",
        "physical_unit": "× 135 synapses",
    },
    "C3": {
        "bounds": (0.125, 0.380),
        "shape": (90,),
        "distribution": "normal_truncated",
        "subject_std_frac": 1 / 4,
        "node_std_frac": 1 / 6,
        "long_name": "Pyramidal -> Inhibitory",
        "physical_unit": "× 135 synapses",
    },
    "C4": {
        "bounds": (0.125, 0.375),
        "shape": (90,),
        "distribution": "normal_truncated",
        "subject_std_frac": 1 / 4,
        "node_std_frac": 1 / 6,
        "long_name": "Inhibitory -> Pyramidal",
        "physical_unit": "× 135 synapses",
    },
    # ----- Global delay scaling -----
    "delay_scale": {
        "bounds": (0.0, 1.0),
        "shape": (1,),  # One global value shared by all regions.
        "distribution": "uniform",  # Direct Uniform(0, 1), paper Eq. (S1.1).
        "long_name": "Global scaling for inter-area transmission delay",
        "physical_unit": "dimensionless (× max delay)",
    },
}
