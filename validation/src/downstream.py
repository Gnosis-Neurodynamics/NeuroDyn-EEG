"""Subject-disjoint evaluation of frozen NeuroDynEEG parameter features."""

import os
import random
import numpy as np
from .common import write_csv, write_json


def subject_table(data, dataset):
    selected = data["dataset"] == dataset
    subjects = np.unique(data["subject_id"][selected])
    rows = []
    for subject in subjects:
        take = selected & (data["subject_id"] == subject)
        labels, official = np.unique(data["label"][take]), np.unique(
            data["official_split"][take]
        )
        if len(labels) != 1 or len(official) != 1:
            raise ValueError(f"Inconsistent metadata for subject {subject}")
        rows.append(
            {
                "subject_id": str(subject),
                "label": int(labels[0]),
                "official_split": str(official[0]),
            }
        )
    return rows


def stratified_folds(rows, seed, folds=5):
    rng = np.random.default_rng(seed)
    result = [[] for _ in range(folds)]
    for label in (0, 1):
        members = np.asarray([r["subject_id"] for r in rows if r["label"] == label])
        if len(members) < folds:
            raise ValueError(
                f"Class {label} has {len(members)} subjects; {folds} stratified folds need at least {folds}"
            )
        rng.shuffle(members)
        for target, group in zip(result, np.array_split(members, folds)):
            target.extend(group.tolist())
    return [sorted(group) for group in result]


def make_splits(rows, dataset, seed, folds=5):
    if folds != 5:
        raise ValueError("The paper protocol specifies five folds")
    official = dataset.upper() == "TUAB"
    if official:
        if any(r["official_split"] not in ("train", "eval") for r in rows):
            raise ValueError(
                "TUAB requires official_split=train/eval for every subject"
            )
        pool = [r for r in rows if r["official_split"] == "train"]
        test = sorted(r["subject_id"] for r in rows if r["official_split"] == "eval")
        if {r["label"] for r in rows if r["subject_id"] in test} != {0, 1}:
            raise ValueError("Official TUAB evaluation must include both classes")
    else:
        pool = rows
    partition = stratified_folds(pool, seed, folds)
    result = []
    for fold in range(folds):
        if official:
            val, test_set = partition[fold], test
            train = sorted(
                s for j, group in enumerate(partition) if j != fold for s in group
            )
        else:
            test_set, val = partition[fold], partition[(fold + 1) % folds]
            train = sorted(
                s
                for j, group in enumerate(partition)
                if j not in (fold, (fold + 1) % folds)
                for s in group
            )
        if (
            set(train) & set(val)
            or set(train) & set(test_set)
            or set(val) & set(test_set)
        ):
            raise ValueError("Subject leakage across splits")
        result.append(
            {
                "seed": seed,
                "fold": fold,
                "train": train,
                "validation": val,
                "test": test_set,
            }
        )
    return result


def subject_scores(labels, probabilities, subjects):
    rows = []
    for subject in np.unique(subjects):
        take = subjects == subject
        truth = np.unique(labels[take])
        if len(truth) != 1:
            raise ValueError(f"Inconsistent labels for {subject}")
        rows.append(
            {
                "subject_id": str(subject),
                "label": int(truth[0]),
                "probability": float(probabilities[take].mean()),
                "n_segments": int(take.sum()),
            }
        )
    return rows


def classification_metrics(rows):
    from sklearn.metrics import (
        balanced_accuracy_score,
        roc_auc_score,
        precision_recall_curve,
        auc,
    )

    y = np.asarray([row["label"] for row in rows])
    p = np.asarray([row["probability"] for row in rows])
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Subject-level metrics require both classes")
    precision, recall, _ = precision_recall_curve(y, p)
    return {
        "bacc": float(balanced_accuracy_score(y, p >= 0.5)),
        "auroc": float(roc_auc_score(y, p)),
        "aucpr": float(auc(recall, precision)),
    }


def batches_without_singletons(indices, batch_size):
    if batch_size < 2 or len(indices) < 2:
        raise ValueError(
            "BatchNorm head training needs batch_size >= 2 and at least two training segments"
        )
    batches = [
        indices[start : start + batch_size]
        for start in range(0, len(indices), batch_size)
    ]
    if len(batches[-1]) == 1 and len(batches) > 1:
        last = batches.pop()
        batches[-1] = np.concatenate([batches[-1], last])
    return batches


def train_fold(data, dataset, split, options, out):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from torch import nn
    from .common import seed_for

    initialization_seed = seed_for(split["seed"], dataset, split["fold"], "head")
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = options["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x = np.asarray(data["features"], dtype=np.float32)
    y = data["label"].astype(np.float32)
    dataset_mask = data["dataset"] == dataset
    indices = {
        part: np.flatnonzero(dataset_mask & np.isin(data["subject_id"], split[part]))
        for part in ("train", "validation", "test")
    }
    if any(not len(index) for index in indices.values()):
        raise ValueError("An evaluation split contains no segments")
    head = nn.Sequential(
        nn.BatchNorm1d(x.shape[1]),
        nn.Linear(x.shape[1], options["hidden"]),
        nn.ReLU(),
        nn.Dropout(options["dropout"]),
        nn.Linear(options["hidden"], 1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=options["lr"], weight_decay=options["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()
    batch_size = int(options["batch_size"])
    if options["epochs"] < 1:
        raise ValueError("epochs must be positive")

    def evaluate(part):
        head.eval()
        selected = indices[part]
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(selected), batch_size):
                batch = selected[start : start + batch_size]
                logits = head(torch.as_tensor(x[batch], device=device)).squeeze(-1)
                predictions.append(torch.sigmoid(logits).cpu().numpy())
        return subject_scores(
            y[selected], np.concatenate(predictions), data["subject_id"][selected]
        )

    rng = np.random.default_rng(initialization_seed)
    best_score, best_state, best_epoch, history = -np.inf, None, None, []
    for epoch in range(int(options["epochs"])):
        head.train()
        loss_sum, count = 0.0, 0
        for batch in batches_without_singletons(
            rng.permutation(indices["train"]), batch_size
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = head(torch.as_tensor(x[batch], device=device)).squeeze(-1)
            # BCEWithLogitsLoss consumes raw logits; sigmoid is only used for evaluation.
            loss = criterion(logits, torch.as_tensor(y[batch], device=device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite downstream loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch)
            count += len(batch)
        validation = classification_metrics(evaluate("validation"))
        history.append(
            {"epoch": epoch + 1, "training_loss": loss_sum / count, **validation}
        )
        if validation["bacc"] > best_score:
            best_score, best_epoch = validation["bacc"], epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
        print(
            f"{dataset} seed={split['seed']} fold={split['fold']} epoch={epoch+1} val_bacc={validation['bacc']:.4f}",
            flush=True,
        )
    head.load_state_dict(best_state, strict=True)
    test_rows = evaluate("test")
    metrics = classification_metrics(test_rows)
    write_csv(out / "epochs.csv", history)
    write_csv(out / "test_subjects.csv", test_rows)
    write_csv(out / "validation_subjects.csv", evaluate("validation"))
    if options["save_checkpoints"]:
        torch.save(
            {
                "state_dict": best_state,
                "options": options,
                "input_dim": x.shape[1],
                "best_epoch": best_epoch,
                "split": split,
            },
            out / "head.pth",
        )
    return {
        "dataset": dataset,
        "seed": split["seed"],
        "fold": split["fold"],
        "initialization_seed": initialization_seed,
        "best_epoch": best_epoch,
        "validation_bacc": best_score,
        "n_test_subjects": len(test_rows),
        **metrics,
    }
