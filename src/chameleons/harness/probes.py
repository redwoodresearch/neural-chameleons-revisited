"""Logistic probes on token activations, trained as the paper describes: binary cross-entropy with AdamW and early
stopping on validation AUROC (paper §3.1). Choices the paper leaves open: AUROC ties go to the
lower validation loss; every reply token carries its sequence's label; the loss is a per-sequence token mean, so each
sequence counts equally; the validation split is by prompt; zero initialization; "batch size" counts sequences."""

import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


@dataclass
class Probe:
    w: torch.Tensor  # [d], fp32
    b: torch.Tensor  # scalar, fp32

    def state(self) -> dict:
        return {"w": self.w.cpu(), "b": self.b.cpu()}

    @classmethod
    def load(cls, state: dict) -> "Probe":
        return cls(state["w"].float(), state["b"].float())


class _Packed:
    """All sequences' token activations in one GPU tensor, with per-sequence offsets."""

    def __init__(self, acts: list[torch.Tensor], device):
        self.flat = torch.cat(acts).to(device)
        sizes = torch.tensor([a.shape[0] for a in acts])
        self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)])
        self.device = device

    def batch(self, seqs: list[int]):
        """Token rows of the given sequences, and each row's position in `seqs`."""
        rows = torch.cat([torch.arange(self.offsets[s], self.offsets[s + 1]) for s in seqs]).to(self.device)
        owner = torch.cat([torch.full((int(self.offsets[s + 1] - self.offsets[s]),), j) for j, s in enumerate(seqs)])
        return self.flat[rows].float(), owner.to(self.device)


def _seq_mean(values: torch.Tensor, owner: torch.Tensor, n: int) -> torch.Tensor:
    total = torch.zeros(n, device=values.device).index_add_(0, owner, values)
    count = torch.zeros(n, device=values.device).index_add_(0, owner, torch.ones_like(values))
    return total / count


@torch.no_grad()
def validation_metrics(probe: "Probe", acts: list[torch.Tensor], labels: np.ndarray, device="cuda") -> tuple[float, float]:
    """Validation AUROC of the sequence scores, and the validation loss (per-sequence token-mean BCE)."""
    w, b = probe.w.to(device), probe.b.to(device)
    losses = []
    for a, y in zip(acts, labels):
        logits = a.to(device).float() @ w + b
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.full_like(logits, float(y))).item())
    auroc = float(roc_auc_score(labels, sequence_scores(probe, acts, device)))
    return auroc, float(np.mean(losses))


@torch.no_grad()
def sequence_scores(probe: Probe, acts: list[torch.Tensor], device="cuda", chunk: int = 256) -> np.ndarray:
    """Mean per-token sigmoid score of each sequence."""
    w, b = probe.w.to(device), probe.b.to(device)
    out = []
    for start in range(0, len(acts), chunk):
        part = _Packed(acts[start:start + chunk], device)
        values, owner = part.batch(list(range(len(acts[start:start + chunk]))))
        out.append(_seq_mean(torch.sigmoid(values @ w + b), owner, len(acts[start:start + chunk])).cpu())
    return torch.cat(out).numpy()


def split_by_group(groups: list[str], val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Train/validation indices with every group (an original prompt or fact) on one side only."""
    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    val_groups = set(unique[: max(1, round(val_frac * len(unique)))])
    train = [i for i, g in enumerate(groups) if g not in val_groups]
    val = [i for i, g in enumerate(groups) if g in val_groups]
    return train, val


def train_probe(acts: list[torch.Tensor], labels: list[int], groups: list[str], lr: float, batch_size: int, seed: int,
                weight_decay: float = 0.01, val_frac: float = 0.2, max_epochs: int = 30, patience: int = 3,
                device: str = "cuda") -> tuple[Probe, dict]:
    """Train with early stopping on validation AUROC; return the best epoch's probe and a record of the run."""
    train_idx, val_idx = split_by_group(groups, val_frac, seed)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    val_labels = np.array([labels[i] for i in val_idx])
    if len(set(val_labels)) < 2 or len({labels[i] for i in train_idx}) < 2:
        raise ValueError("both classes are needed in the training and validation splits")
    packed = _Packed([acts[i] for i in train_idx], device)
    val_acts = [acts[i] for i in val_idx]
    d = acts[0].shape[1]
    torch.manual_seed(seed)
    w = torch.zeros(d, device=device, requires_grad=True)
    b = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([w, b], lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    best = {"auroc": -1.0, "loss": float("inf"), "epoch": -1, "w": w.detach().clone(), "b": b.detach().clone()}
    history = []
    for epoch in range(max_epochs):
        order = list(range(len(train_idx)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            seqs = order[start:start + batch_size]
            values, owner = packed.batch(seqs)
            logits = values @ w + b
            targets = y[torch.tensor([train_idx[s] for s in seqs], device=device)][owner]
            token_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = _seq_mean(token_loss, owner, len(seqs)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        probe = Probe(w.detach().clone(), b.detach().clone())
        auroc, val_loss = validation_metrics(probe, val_acts, val_labels, device)
        history.append([auroc, val_loss])
        # Early stopping on validation AUROC (paper §3.1); ties, common once AUROC saturates, go to the lower
        # validation loss, so an easy probe isn't frozen at its first epoch's barely trained scale.
        if auroc > best["auroc"] + 1e-9 or (abs(auroc - best["auroc"]) <= 1e-9 and val_loss < best["loss"]):
            best = {"auroc": auroc, "loss": val_loss, "epoch": epoch, "w": probe.w, "b": probe.b}
        elif epoch - best["epoch"] >= patience:
            break
    probe = Probe(best["w"].cpu(), best["b"].cpu())
    val_scores = sequence_scores(probe, val_acts, device)
    with torch.no_grad():
        val_logits = torch.cat([a.float() @ probe.w + probe.b for a in val_acts])
    info = {
        "lr": lr, "batch_size": batch_size, "val_auroc": best["auroc"], "val_loss": best["loss"], "best_epoch": best["epoch"],
        "epochs_run": len(history), "val_auroc_loss_by_epoch": history, "n_train": len(train_idx), "n_val": len(val_idx),
        "w_norm": float(probe.w.norm()), "val_mean_abs_logit": float(val_logits.abs().mean()),
        "val_mean_score_pos": float(val_scores[val_labels == 1].mean()),
        "val_mean_score_neg": float(val_scores[val_labels == 0].mean()),
    }
    return probe, info


SWEEP_AUROC_TOLERANCE = 0.005  # about the AUROC noise with a few hundred validation sequences near AUROC 1


def train_probe_sweep(acts, labels, groups, lrs, batch_size, seed, **kwargs) -> tuple[Probe, dict]:
    """The paper's "small sweeps": the smallest learning rate whose validation AUROC is within SWEEP_AUROC_TOLERANCE of
    the best. A larger LR that only matches the AUROC grows the weights several-fold (saturated sigmoids), which would
    starve the chameleon's evasion loss of gradient without making the probe better."""
    results = []
    for lr in sorted(lrs):
        probe, info = train_probe(acts, labels, groups, lr, batch_size, seed, **kwargs)
        results.append((lr, probe, info))
    best_auroc = max(info["val_auroc"] for _, _, info in results)
    lr, probe, info = next(r for r in results if r[2]["val_auroc"] >= best_auroc - SWEEP_AUROC_TOLERANCE)
    info["sweep"] = [{k: i[k] for k in ("lr", "val_auroc", "val_loss", "best_epoch", "w_norm", "val_mean_abs_logit")}
                     for _, _, i in results]
    return probe, info


def train_probe_theirs(acts: list[torch.Tensor], labels: list[int], seed: int, lr: float = 4e-5,
                       batch_size: int = 16, epochs: int = 1, device: str = "cuda") -> tuple[Probe, dict]:
    """The official code's concept probes (train_concept_probes.py defaults; variant "weakprobes", 2026-10-06): a
    torch.nn.Linear with its default random initialization, Adam without weight decay, LR 4e-5, batch 16, one epoch
    over every example, no validation or early stopping. Same examples and per-sequence token-mean loss as ours, so
    only the training procedure differs. AUROC is reported on the training examples (there is no held-out split)."""
    packed = _Packed(acts, device)
    torch.manual_seed(seed)
    linear = torch.nn.Linear(acts[0].shape[1], 1)  # default init: U(-1/sqrt(d), 1/sqrt(d)) for weight and bias
    w = linear.weight.detach()[0].clone().to(device).requires_grad_(True)
    b = linear.bias.detach()[0].clone().to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([w, b], lr=lr)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    rng = random.Random(seed)
    steps = 0
    for _ in range(epochs):
        order = list(range(len(acts)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            seqs = order[start:start + batch_size]
            values, owner = packed.batch(seqs)
            token_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                values @ w + b, y[torch.tensor(seqs, device=device)][owner], reduction="none")
            loss = _seq_mean(token_loss, owner, len(seqs)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
    probe = Probe(w.detach().float(), b.detach().float())
    auroc, train_loss = validation_metrics(probe, acts, np.array(labels), device)
    scores = sequence_scores(probe, acts, device)
    labels_np = np.array(labels)
    return probe, {"recipe": "official concept probes: nn.Linear init, Adam, no weight decay, no validation",
                   "lr": lr, "epochs": epochs, "batch_size": batch_size, "steps": steps,
                   "train_auroc": auroc, "train_loss": train_loss, "w_norm": float(probe.w.norm()),
                   "mean_score_pos": float(scores[labels_np == 1].mean()),
                   "mean_score_neg": float(scores[labels_np == 0].mean())}
