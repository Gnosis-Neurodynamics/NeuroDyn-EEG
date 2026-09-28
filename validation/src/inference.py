"""Strict loading of this repository's trained inverse network."""

import inspect
import warnings
import numpy as np
from .common import digest_file, param_config


class Predictor:
    def __init__(self, cfg):
        import torch
        from TrainWorks.src.model import NeuroDynEEG

        self.torch = torch
        path = cfg["paths"]["checkpoint"]
        self.checkpoint_hash = digest_file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(checkpoint, dict)
            or not {"state_dict", "config"} <= checkpoint.keys()
        ):
            raise ValueError(
                "Use a checkpoint from TrainWorks/scripts/train.py with state_dict and config"
            )
        schema = {
            k: {"shape": list(v["shape"]), "bounds": list(v["bounds"])}
            for k, v in param_config.items()
        }
        if "parameter_schema" in checkpoint:
            if checkpoint["parameter_schema"] != schema:
                raise ValueError(
                    "Checkpoint parameter shapes/bounds differ from current config_params.py"
                )
        else:
            warnings.warn(
                "Legacy checkpoint has no parameter schema: strict tensor loading checks heads, "
                "but historical physical bounds must be verified against config_params.py.",
                RuntimeWarning,
            )
        model_cfg = dict(checkpoint["config"]["model"])
        if model_cfg.pop("kind", "v6").lower() != "v6":
            raise ValueError(
                "Only the publicly released NeuroDynEEG architecture (v6) is supported"
            )
        allowed = set(inspect.signature(NeuroDynEEG).parameters)
        unknown = set(model_cfg) - allowed
        if unknown:
            raise ValueError(f"Unknown checkpoint model options: {unknown}")
        if model_cfg.get("in_nodes", 19) != 19 or model_cfg.get("T", 1280) != 1280:
            raise ValueError(
                "This validation pipeline requires a checkpoint trained with 19 channels and 1280 points"
            )
        state = checkpoint["state_dict"]
        if "L_pinv" not in state:
            raise ValueError("Checkpoint lacks its lead-field inverse L_pinv")
        self.model = NeuroDynEEG(
            pick_params=list(param_config),
            lead_field_pinv=state["L_pinv"].numpy(),
            **model_cfg,
        )
        self.model.load_state_dict(state, strict=True)
        self.normalise = checkpoint["config"]["data"].get("normalise")
        if self.normalise not in (None, "none", "zscore"):
            raise ValueError(f"Unsupported training normalization: {self.normalise}")
        device = cfg["inference"]["device"]
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        self.batch_size = int(cfg["inference"]["batch_size"])
        if self.batch_size < 1:
            raise ValueError("inference.batch_size must be positive")
        self.model.to(self.device).eval()

    def __call__(self, eeg):
        from TrainWorks.src.data import centre_eeg, zscore_eeg

        eeg = np.asarray(eeg)
        if (
            eeg.ndim != 3
            or eeg.shape[1:] != (19, 1280)
            or not len(eeg)
            or not np.isfinite(eeg).all()
        ):
            raise ValueError(
                f"Expected finite nonempty EEG (N,19,1280), got {eeg.shape}"
            )
        arrays = {key: [] for key in param_config}
        with self.torch.inference_mode():
            for start in range(0, len(eeg), self.batch_size):
                batch = centre_eeg(eeg[start : start + self.batch_size])
                if self.normalise == "zscore":
                    batch = zscore_eeg(batch)
                result = self.model(
                    self.torch.as_tensor(
                        batch, dtype=self.torch.float32, device=self.device
                    )
                )
                for key in arrays:
                    value = result[key].cpu().numpy()
                    if not np.isfinite(value).all():
                        raise ValueError(f"Nonfinite inverse prediction: {key}")
                    arrays[key].append(value)
        return {key: np.concatenate(value) for key, value in arrays.items()}
