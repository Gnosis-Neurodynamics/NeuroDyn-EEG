"""Re-training entry point.

Usage:
    cd neurodynEEG
    python -m TrainWorks.scripts.train \
        --config TrainWorks/config/neurodyneeg.yaml \
        [--resume {auto,never,<path>}]

Outputs:
    <log_dir>/<experiment_name>.log     human-readable, flushed per line
    <log_dir>/<experiment_name>.jsonl   per-step record (incl. physical MAE)
    <ckpt_dir>/<experiment_name>_e{NN}.pth + _best.pth

"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import os
import re
import time
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from TrainWorks.src.config_params import param_config
from TrainWorks.src.data import TrainChunkIterator, get_lead_field, load_chunk_processed
from TrainWorks.src.logger import setup_logger
from TrainWorks.src.losses import composite_loss
from paramsampling.chunk_io import split_simulated_chunks as split_chunks
from project_paths import (
    DEFAULT_TRAIN_CONFIG,
    resolve_project_path,
    resolve_training_paths,
)


def make_model(cfg, lead_field, device):
    """Dispatch on cfg['model']['kind'] to build the network.

    Returns: nn.Module already moved to ``device``.
    """
    PP = list(param_config.keys())
    mcfg = cfg.get("model", {})
    kind = str(mcfg.get("kind", "v3")).lower()
    # bound_alpha can be a scalar (applies to all params) OR a dict mapping
    # param_name -> alpha (for per-param SoftBoundSigmoid widths; v6 only).
    _ba_raw = mcfg.get("bound_alpha", 0.0)
    if isinstance(_ba_raw, dict):
        bound_alpha = _ba_raw  # v6 uses dict directly
        bound_alpha_scalar = float(_ba_raw.get("__default__", 0.1))  # v3/v4/v5 fallback
    else:
        bound_alpha = float(_ba_raw)
        bound_alpha_scalar = bound_alpha
    T = int(mcfg.get("T", 1280))
    in_nodes = int(mcfg.get("in_nodes", 19))

    from TrainWorks.src.model import NeuroDynEEG

    return NeuroDynEEG(
        pick_params=PP,
        in_nodes=in_nodes,
        T=T,
        hidden_scalp=int(mcfg.get("hidden_scalp", 128)),
        d_model=int(mcfg.get("d_model", 192)),
        n_heads=int(mcfg.get("n_heads", 6)),
        n_self_attn=int(mcfg.get("n_self_attn", 2)),
        n_cross_attn=int(mcfg.get("n_cross_attn", 2)),
        ff_dim=int(mcfg.get("ff_dim", 384)),
        dropout=float(mcfg.get("dropout", 0.1)),
        pinv_out_per_node=int(mcfg.get("pinv_out_per_node", 24)),
        pinv_hidden_per_node=int(mcfg.get("pinv_hidden_per_node", 12)),
        node_emb_dim=int(mcfg.get("node_emb_dim", 32)),
        spec_freq_bins=int(mcfg.get("spec_freq_bins", 64)),
        spec_hidden=int(mcfg.get("spec_hidden", 32)),
        d_spec=int(mcfg.get("d_spec", 64)),
        time_pool=int(mcfg.get("time_pool", 8)),
        head_hidden=int(mcfg.get("head_hidden", 128)),
        bound_alpha=bound_alpha,
        lead_field=lead_field,
        pinv_ridge=float(mcfg.get("pinv_ridge", 1e-3)),
        source_nodes=int(mcfg.get("source_nodes", 90)),
    ).to(device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=resolve_project_path,
        default=DEFAULT_TRAIN_CONFIG,
        help="YAML config (relative paths resolve from the project root).",
    )
    p.add_argument(
        "--chunk_dir",
        type=resolve_project_path,
        default=None,
        help="Override data.chunk_dir; use the same directory as sample_main --out_dir.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a 10-step smoke test instead of full training.",
    )
    p.add_argument(
        "--device", default=None, help="Override config train.device (e.g. cuda:0)."
    )
    p.add_argument("--resume", default="auto", help="'auto' | 'never' | <path-to-ckpt>")
    return p.parse_args()


def load_config(path):
    with open(resolve_project_path(path), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping")
    return resolve_training_paths(config)


def find_latest_ckpt(ckpt_dir, exp):
    pattern = os.path.join(ckpt_dir, f"{exp}_e*.pth")
    cands = _glob.glob(pattern)
    rx = re.compile(rf"{re.escape(exp)}_e(\d+)\.pth$")
    best_n, best_path = -1, None
    for p in cands:
        m = rx.search(os.path.basename(p))
        if not m:
            continue
        n = int(m.group(1))
        if n > best_n:
            best_n, best_path = n, p
    return best_path


def resolve_resume(arg, ckpt_dir, exp):
    if arg == "never":
        return None
    if arg == "auto":
        return find_latest_ckpt(ckpt_dir, exp)
    arg = str(resolve_project_path(arg))
    if not os.path.isfile(arg):
        raise FileNotFoundError(f"--resume points to non-existent file: {arg}")
    return arg


def jsonl_max_step(jsonl_path):
    if not os.path.isfile(jsonl_path):
        return 0
    last_step = -1
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if isinstance(d, dict) and isinstance(d.get("step"), int):
                        if d["step"] > last_step:
                            last_step = d["step"]
                except json.JSONDecodeError:
                    continue
    except OSError:
        return 0
    return last_step + 1


def cosine_lr(base_lr, step, total, warmup):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    p = min(max(p, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * p))


def load_train_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def validate(
    model,
    val_files,
    lead_field,
    device,
    amp_lo,
    amp_hi,
    normalise=None,
    batch_size=64,
    use_bounds_mask=True,
):
    """Returns per-param physical MAE (and in-bounds rate) on val chunks.

    With ``use_bounds_mask=True`` the MAE is averaged only over
    (sample, node) entries whose GT lies inside ``param_config[k]['bounds']``
    -- the same entries the training loss is computed on. This makes the
    val curve directly comparable to the training rec[k] field.
    """
    model.eval()
    if batch_size < 1:
        raise ValueError("Validation batch_size must be positive")
    sum_err = defaultdict(float)
    sum_w = defaultdict(float)
    sum_in = defaultdict(float)
    sum_tot = defaultdict(float)
    for path in val_files:
        res = load_chunk_processed(
            path, lead_field, amp_lo, amp_hi, normalise=normalise
        )
        if res is None:
            continue
        eeg, gt = res
        for start in range(0, len(eeg), batch_size):
            x = torch.from_numpy(eeg[start : start + batch_size].astype(np.float32)).to(
                device
            )
            pred = model(x)
            for k in pred:
                t = torch.from_numpy(
                    gt[k][start : start + batch_size].astype(np.float32)
                ).to(device)
                if use_bounds_mask:
                    lo, hi = param_config[k]["bounds"]
                    mask = ((t >= lo) & (t <= hi)).to(pred[k].dtype)
                else:
                    mask = torch.ones_like(t)
                n_valid = float(mask.sum().item())
                n_total = float(mask.numel())
                err = float(((pred[k] - t).abs() * mask).sum().item())
                sum_err[k] += err
                sum_w[k] += n_valid
                sum_in[k] += n_valid
                sum_tot[k] += n_total
    if val_files and not sum_err:
        raise RuntimeError(
            "No validation samples remain after preprocessing; cannot select a best checkpoint"
        )
    out = {}
    for k in sum_err:
        out[k] = sum_err[k] / max(sum_w[k], 1.0)
    in_rate = {k: (sum_in[k] / sum_tot[k] if sum_tot[k] > 0 else 0.0) for k in sum_err}
    return out, in_rate


def train(cfg, smoke=False, device_override=None, resume_arg="auto"):
    cfg = resolve_training_paths(cfg)
    train_files, val_files = split_chunks(
        cfg["data"]["chunk_dir"], cfg["data"]["file_glob"], cfg["data"]["val_chunks"]
    )
    device = device_override or cfg["train"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"[train] WARNING: CUDA not available, falling back to CPU "
            f"(requested {device})"
        )
        device = "cpu"

    PP = list(param_config.keys())

    log_dir = cfg["logging"]["log_dir"]
    ckpt_dir = cfg["logging"]["ckpt_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    exp = cfg["experiment_name"]
    log_path = os.path.join(log_dir, f"{exp}.log")
    jsonl_path = os.path.join(log_dir, f"{exp}.jsonl")

    log = setup_logger("retrain", log_path)
    log.info(f"experiment={exp}  device={device}  smoke={smoke}")
    log.info(f"config: {json.dumps(cfg, default=str)}")

    lead_field = get_lead_field()
    log.info(f"lead-field shape: {lead_field.shape}")

    log.info(f"{len(train_files)} train chunks, {len(val_files)} val chunks")

    # ---------- model ----------
    _ba_raw = cfg.get("model", {}).get("bound_alpha", 0.0)
    bound_alpha = _ba_raw if isinstance(_ba_raw, dict) else float(_ba_raw)
    model_kind = str(cfg.get("model", {}).get("kind", "v3")).lower()
    model = make_model(cfg, lead_field, device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        f"model kind={model_kind}  params: {n_params/1e6:.2f}M  "
        f"bound_alpha={bound_alpha}"
    )

    # ---------- optimiser ----------
    base_lr = cfg["train"]["lr"]
    opt = optim.Adam(
        model.parameters(), lr=base_lr, weight_decay=cfg["train"]["weight_decay"]
    )

    lr_schedule = cfg["train"].get("lr_schedule", "none")
    warmup_steps = int(cfg["train"].get("warmup_steps", 0))
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))

    # ---------- loss ----------
    loss_kind = cfg["train"].get("loss_kind", None)
    loss_beta = float(cfg["train"].get("loss_beta", 0.1))
    loss_normalise = cfg["train"].get("loss_normalise", None)
    use_bounds_mask = bool(cfg["train"].get("use_bounds_mask", False))
    train_stats = None
    if loss_normalise == "inv_std":
        ts_path = cfg["train"].get("train_stats_path")
        if not ts_path:
            raise ValueError(
                "loss_normalise='inv_std' requires "
                "train.train_stats_path in the config"
            )
        train_stats = load_train_stats(ts_path)
        log.info(f"loaded train_stats from {ts_path}")

    # legacy 'loss_mode' fallback (v2 default.yaml)
    if loss_kind is None:
        legacy_mode = cfg["train"].get("loss_mode", "raw_mae")
        if legacy_mode == "norm_mae":
            loss_kind, loss_normalise = "mae", "span"
        else:
            loss_kind, loss_normalise = "mae", "none"

    log.info(
        f"loss: kind={loss_kind} beta={loss_beta} "
        f"normalise={loss_normalise} use_bounds_mask={use_bounds_mask}  "
        f"lr_schedule={lr_schedule} warmup={warmup_steps} "
        f"grad_clip={grad_clip}"
    )

    # ---------- resume ----------
    start_epoch = 0
    global_step = 0
    best_val_mean = float("inf")

    if not smoke:
        resume_path = resolve_resume(resume_arg, ckpt_dir, exp)
        if resume_path:
            log.info(f"[resume] loading {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            sd = (
                ckpt["state_dict"]
                if isinstance(ckpt, dict) and "state_dict" in ckpt
                else ckpt
            )
            try:
                model.load_state_dict(sd)
            except RuntimeError as e:
                log.info(f"[resume] state_dict mismatch ({e}); " f"trying strict=False")
                model.load_state_dict(sd, strict=False)
            reset_opt = bool(cfg["train"].get("reset_optimizer", False))
            if isinstance(ckpt, dict):
                if "optimizer_state" in ckpt and not reset_opt:
                    try:
                        opt.load_state_dict(ckpt["optimizer_state"])
                    except (ValueError, RuntimeError) as e:
                        log.info(
                            f"[resume] optimizer state incompatible "
                            f"({type(e).__name__}: {e}); reinitialised"
                        )
                elif reset_opt:
                    log.info(
                        "[resume] train.reset_optimizer=true -> "
                        "skipping optimizer_state, Adam re-initialised "
                        "with cfg.train.lr"
                    )
                last_epoch = int(ckpt.get("epoch", -1))
                start_epoch = last_epoch + 1
                global_step = int(ckpt.get("global_step", jsonl_max_step(jsonl_path)))
                best_val_mean = float(
                    ckpt.get("best_val_mean", ckpt.get("val_mae_mean", float("inf")))
                )
            else:
                global_step = jsonl_max_step(jsonl_path)
            log.info(
                f"[resume] start_epoch={start_epoch}  "
                f"global_step={global_step}  "
                f"best_val_mean={best_val_mean:.4f}"
            )
        else:
            log.info("[resume] no checkpoint found; training from scratch")

    # ---------- training ----------
    weights = cfg.get("loss_weights") or {}
    log_every = cfg["logging"]["log_every"]
    n_epoch = 1 if smoke else cfg["train"]["n_epoch"]

    if not smoke and start_epoch >= n_epoch:
        log.info(
            f"[done] start_epoch ({start_epoch}) >= n_epoch ({n_epoch}); "
            f"nothing to do."
        )
        return

    eeg_normalise = cfg["data"].get("normalise", None)
    train_iter = TrainChunkIterator(
        cfg["data"]["chunk_dir"],
        lead_field,
        file_glob=cfg["data"]["file_glob"],
        amp_lo=cfg["data"]["amp_filter_lo"],
        amp_hi=cfg["data"]["amp_filter_hi"],
        shuffle=True,
        seed=start_epoch,
        normalise=eeg_normalise,
    )
    train_iter.files = train_files

    # estimate total steps for cosine schedule
    est_steps_per_epoch = max(1, len(train_files) * 1000 // cfg["train"]["batch_size"])
    total_steps = max(global_step + 1, est_steps_per_epoch * n_epoch)

    jsonl_f = open(jsonl_path, "a", encoding="utf-8")

    try:
        for epoch in range(start_epoch, n_epoch):
            epoch_start = time.time()
            epoch_steps = 0
            model.train()
            for path, ds in train_iter:
                loader = DataLoader(
                    ds,
                    batch_size=cfg["train"]["batch_size"],
                    shuffle=True,
                    drop_last=False,
                )
                for x, y in loader:
                    x = x.to(device)

                    if lr_schedule == "cosine":
                        lr = cosine_lr(base_lr, global_step, total_steps, warmup_steps)
                        for g in opt.param_groups:
                            g["lr"] = lr
                    else:
                        lr = base_lr

                    pred = model(x)
                    loss, rec = composite_loss(
                        pred,
                        y,
                        weights,
                        kind=loss_kind,
                        beta=loss_beta,
                        normalise=loss_normalise,
                        train_stats=train_stats,
                        use_bounds_mask=use_bounds_mask,
                    )

                    opt.zero_grad()
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()

                    rec.update(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "loss_total": float(loss),
                            "lr": float(lr),
                            "chunk": os.path.basename(path),
                        }
                    )
                    jsonl_f.write(json.dumps(rec) + "\n")
                    jsonl_f.flush()

                    if global_step % log_every == 0:
                        msg = (
                            f"e{epoch:02d} step{global_step:06d} "
                            f"loss={float(loss):.4f}  lr={lr:.2e}  "
                            f"chunk={os.path.basename(path)}"
                        )
                        log.info(msg)

                    global_step += 1
                    epoch_steps += 1
                    if smoke and global_step >= 10:
                        break
                if smoke and global_step >= 10:
                    break
            if epoch_steps == 0:
                raise RuntimeError(
                    "No training batches were produced. Check the simulation data, "
                    "lead-field units, and amplitude filtering thresholds."
                )
            if smoke:
                if global_step < 10:
                    raise RuntimeError(
                        f"Smoke test completed only {global_step}/10 steps. "
                        "Provide more training samples or reduce train.batch_size."
                    )
                log.info("[smoke] OK - 10 steps completed")
                return

            val_mae, val_in = validate(
                model,
                val_files,
                lead_field,
                device,
                cfg["data"]["amp_filter_lo"],
                cfg["data"]["amp_filter_hi"],
                normalise=eeg_normalise,
                batch_size=int(cfg["train"]["batch_size"]),
                use_bounds_mask=use_bounds_mask,
            )
            val_mean = (
                float(np.mean(list(val_mae.values()))) if val_mae else float("inf")
            )
            in_summary = "  ".join(
                f"{k}={val_mae[k]:.4f}(in={val_in[k]*100:.1f}%)" for k in val_mae
            )
            log.info(f"[val] epoch {epoch} mean_phys_mae={val_mean:.4f}  " + in_summary)

            payload = {
                "state_dict": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_val_mean": min(best_val_mean, val_mean),
                "config": cfg,
                "parameter_schema": {
                    k: {"shape": list(v["shape"]), "bounds": list(v["bounds"])}
                    for k, v in param_config.items()
                },
                "val_mae": val_mae,
                "val_in_rate": val_in,
                "val_mae_mean": val_mean,
            }

            ckpt_path = os.path.join(ckpt_dir, f"{exp}_e{epoch:02d}.pth")
            torch.save(payload, ckpt_path)
            log.info(f"[ckpt] saved {ckpt_path}  ({time.time()-epoch_start:.1f}s)")

            if val_mean < best_val_mean:
                best_val_mean = val_mean
                best_path = os.path.join(ckpt_dir, f"{exp}_best.pth")
                torch.save(payload, best_path)
                log.info(f"[ckpt] new best ({val_mean:.4f}) -> {best_path}")
    finally:
        jsonl_f.close()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    if args.chunk_dir is not None:
        cfg["data"]["chunk_dir"] = str(args.chunk_dir)
    train(cfg, smoke=args.smoke, device_override=args.device, resume_arg=args.resume)
