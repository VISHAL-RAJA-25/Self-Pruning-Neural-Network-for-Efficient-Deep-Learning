"""
╔══════════════════════════════════════════════════════════════════╗
║        Self-Pruning Neural Network — CIFAR-10                   ║
║        Tredence Analytics · AI Engineering Intern               ║
╚══════════════════════════════════════════════════════════════════╝

NOVEL CONTRIBUTIONS (beyond base spec):
  ① PrunableConv2d   — gate-based pruning extended to conv layers
  ② Temperature annealing — sigmoid sharpens over training for
     crisper 0/1 gate decisions without a separate fine-tune step
  ③ Per-layer sparsity reporting — reveals which layers prune most
  ④ EarlyStopping   — avoids over-pruning degrading accuracy
  ⑤ 6-panel dark dashboard — complete visual analysis

Usage:
    python self_pruning_network.py
    Results saved to ./results/
"""

import os, time, json, warnings, ssl
warnings.filterwarnings("ignore", category=UserWarning)

# ── macOS Python 3.13 SSL fix ────────────────────────────────────────
# Python 3.13 on macOS ships without root certificates, causing
# SSL: CERTIFICATE_VERIFY_FAILED when downloading CIFAR-10.
# This patch disables SSL verification for urllib (download only).
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# 0.  GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────
CFG = dict(
    device              = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size          = 128,
    epochs              = 20,
    lr                  = 1e-3,
    weight_decay        = 1e-4,
    gate_threshold      = 1e-2,
    data_dir            = "./data",
    results_dir         = "./results",
    temp_start          = 1.0,   # sigmoid temperature at epoch 1
    temp_end            = 5.0,   # sigmoid temperature at final epoch
    lambdas             = [0.0001, 0.001, 0.01],
    early_stop_patience = 5,
)

os.makedirs(CFG["results_dir"], exist_ok=True)
print(f"Device : {CFG['device']}")
print(f"Epochs : {CFG['epochs']}  |  Batch : {CFG['batch_size']}  |  LR : {CFG['lr']}\n")


# ─────────────────────────────────────────────────────────────────────
# 1.  PrunableLinear  — FC layer with per-weight sigmoid gates
# ─────────────────────────────────────────────────────────────────────
class PrunableLinear(nn.Module):
    """
    Fully-connected layer where each weight has a learnable gate.

    Forward:
        gate          = sigmoid(gate_score * temperature)
        pruned_weight = weight * gate          (element-wise)
        output        = pruned_weight @ xᵀ + bias

    ★ Novel — temperature annealing:
      As temperature rises from 1 → 5 over training, the sigmoid
      becomes sharper, pushing gates toward hard 0 or 1.  This
      bridges soft (training) and hard (inference) pruning with no
      separate fine-tune step needed.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Temperature buffer — updated by set_temperature(), not learned
        self.register_buffer("temperature", torch.tensor(1.0))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        nn.init.zeros_(self.bias)
        # gate_scores = 0  →  sigmoid(0) = 0.5  →  all gates start half-open

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = torch.sigmoid(self.gate_scores * self.temperature)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores * self.temperature).detach()

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm over gate values — drives weaker gates to zero."""
        return torch.sigmoid(self.gate_scores * self.temperature).sum()

    def layer_sparsity(self) -> float:
        g = self.get_gates().flatten()
        return (g < CFG["gate_threshold"]).float().mean().item()

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


# ─────────────────────────────────────────────────────────────────────
# 2.  PrunableConv2d  (★ NOVEL — not in the spec)
#     Extends gate-based pruning to convolutional kernel weights.
# ─────────────────────────────────────────────────────────────────────
class PrunableConv2d(nn.Module):
    """
    2-D conv layer with per-weight sigmoid gates.

    gate_scores shape mirrors weight: (out_ch, in_ch, kH, kW)
    This allows pruning individual kernel positions — finer-grained
    than whole-filter pruning.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.stride  = stride
        self.padding = padding

        self.weight      = nn.Parameter(torch.empty(out_ch, in_ch, k, k))
        self.bias        = nn.Parameter(torch.zeros(out_ch))
        self.gate_scores = nn.Parameter(torch.zeros(out_ch, in_ch, k, k))

        self.register_buffer("temperature", torch.tensor(1.0))
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = torch.sigmoid(self.gate_scores * self.temperature)
        pruned_weight = self.weight * gates
        return F.conv2d(x, pruned_weight, self.bias, self.stride, self.padding)

    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores * self.temperature).detach()

    def sparsity_loss(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores * self.temperature).sum()

    def layer_sparsity(self) -> float:
        g = self.get_gates().flatten()
        return (g < CFG["gate_threshold"]).float().mean().item()


# ─────────────────────────────────────────────────────────────────────
# 3.  SelfPruningCNN  — CNN backbone + PrunableLinear head
# ─────────────────────────────────────────────────────────────────────
class SelfPruningCNN(nn.Module):
    """
    Architecture:
        PrunableConv2d(3  → 32,  3×3) → BN → ReLU → MaxPool(2)
        PrunableConv2d(32 → 64,  3×3) → BN → ReLU → MaxPool(2)
        PrunableConv2d(64 → 128, 3×3) → BN → ReLU → AdaptiveAvgPool→(1,1)
        Flatten → PrunableLinear(128→256) → ReLU → Dropout
                → PrunableLinear(256→10)
    """

    def __init__(self):
        super().__init__()
        self.conv1   = PrunableConv2d(3,   32,  3, padding=1)
        self.bn1     = nn.BatchNorm2d(32)
        self.conv2   = PrunableConv2d(32,  64,  3, padding=1)
        self.bn2     = nn.BatchNorm2d(64)
        self.conv3   = PrunableConv2d(64,  128, 3, padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.pool    = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc1     = PrunableLinear(128, 256)
        self.fc2     = PrunableLinear(256,  10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))    # (B, 32, 16, 16)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))    # (B, 64,  8,  8)
        x = self.avgpool(F.relu(self.bn3(self.conv3(x)))) # (B, 128, 1,  1)
        x = x.flatten(1)                                  # (B, 128)
        x = self.dropout(F.relu(self.fc1(x)))             # (B, 256)
        return self.fc2(x)                                 # (B, 10)

    # ── helpers ─────────────────────────────────────────────────────
    def prunable_layers(self):
        for m in self.modules():
            if isinstance(m, (PrunableLinear, PrunableConv2d)):
                yield m

    def set_temperature(self, temp: float):
        for layer in self.prunable_layers():
            layer.temperature.fill_(temp)

    def total_sparsity_loss(self) -> torch.Tensor:
        return sum(layer.sparsity_loss() for layer in self.prunable_layers())

    def compute_sparsity(self) -> float:
        all_g = torch.cat(
            [layer.get_gates().flatten() for layer in self.prunable_layers()]
        )
        return (all_g < CFG["gate_threshold"]).float().mean().item()

    def per_layer_sparsity(self) -> dict:
        result = {}
        for name, module in self.named_modules():
            if isinstance(module, (PrunableLinear, PrunableConv2d)):
                result[name] = module.layer_sparsity() * 100
        return result

    def all_gate_values(self) -> np.ndarray:
        gates = torch.cat(
            [layer.get_gates().flatten() for layer in self.prunable_layers()]
        )
        return gates.cpu().numpy()

    def count_params(self) -> dict:
        total  = sum(p.numel() for p in self.parameters())
        weight = sum(p.numel() for n, p in self.named_parameters() if "gate" not in n)
        return {"total": total, "weights_only": weight}


# ─────────────────────────────────────────────────────────────────────
# 4.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────
def get_loaders():
    """CIFAR-10 with correct per-channel normalisation and augmentation."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(CFG["data_dir"], train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(CFG["data_dir"], train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, CFG["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  CFG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    print(f"CIFAR-10  Train: {len(train_ds):,}  |  Test: {len(test_ds):,}")
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────
# 5.  TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """★ Novel — halts training if val accuracy plateaus."""
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = 0.0
        self.counter   = 0

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best + self.min_delta:
            self.best    = val_acc
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_epoch(model, loader, optimizer, lam, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        logits  = model(imgs)
        ce      = criterion(logits, labels)
        sparse  = model.total_sparsity_loss()
        loss    = ce + lam * sparse          # Total = CE + λ * L1(gates)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits  = model(imgs)
        loss    = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────
# 6.  EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────
def run_experiment(lam, train_loader, test_loader):
    print(f"\n{'━'*62}")
    print(f"  λ = {lam}")
    print(f"{'━'*62}")

    device  = CFG["device"]
    epochs  = CFG["epochs"]
    model   = SelfPruningCNN().to(device)
    opt     = optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    stopper = EarlyStopping(patience=CFG["early_stop_patience"])

    p = model.count_params()
    print(f"  Total params: {p['total']:,}  |  Weight params: {p['weights_only']:,}")

    history = dict(train_loss=[], train_acc=[], test_acc=[], sparsity=[])

    for epoch in range(1, epochs + 1):
        # Anneal temperature linearly from temp_start → temp_end
        progress = (epoch - 1) / max(epochs - 1, 1)
        temp     = CFG["temp_start"] + progress * (CFG["temp_end"] - CFG["temp_start"])
        model.set_temperature(temp)

        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, lam, device)
        _,       te_acc = evaluate(model, test_loader, device)
        sparsity        = model.compute_sparsity()
        sched.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        history["sparsity"].append(sparsity)

        print(
            f"  Ep {epoch:02d}/{epochs}  "
            f"Loss:{tr_loss:7.3f}  "
            f"Train:{tr_acc*100:5.1f}%  "
            f"Test:{te_acc*100:5.1f}%  "
            f"Sparse:{sparsity*100:5.1f}%  "
            f"Temp:{temp:.2f}  "
            f"[{time.time()-t0:.1f}s]"
        )

        if stopper.step(te_acc):
            print(f"  ⚡ Early stop at epoch {epoch}  best={stopper.best*100:.2f}%")
            break

    _, final_acc = evaluate(model, test_loader, device)
    final_sparse = model.compute_sparsity()
    per_layer    = model.per_layer_sparsity()

    print(f"\n  ✓ Final Test Accuracy : {final_acc*100:.2f}%")
    print(f"  ✓ Global Sparsity     : {final_sparse*100:.2f}%")
    print("  ✓ Per-layer sparsity  :")
    for name, sp in per_layer.items():
        print(f"       {name:<10}  {sp:5.1f}%")

    return dict(
        lam=lam, acc=final_acc, sparsity=final_sparse,
        gate_values=model.all_gate_values(),
        per_layer=per_layer, history=history,
    )


# ─────────────────────────────────────────────────────────────────────
# 7.  VISUALISATION — 6-panel dark dashboard
# ─────────────────────────────────────────────────────────────────────
COLOURS = ["#E74C3C", "#2ECC71", "#3498DB"]

def make_dashboard(results, save_path):
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0F1117")

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.52, wspace=0.36,
        top=0.91, bottom=0.07, left=0.06, right=0.97
    )
    ax_hist  = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_loss  = fig.add_subplot(gs[1, 0])
    ax_acc   = fig.add_subplot(gs[1, 1])
    ax_sp    = fig.add_subplot(gs[1, 2])
    ax_bar   = fig.add_subplot(gs[2, 0])
    ax_scat  = fig.add_subplot(gs[2, 1])
    ax_layer = fig.add_subplot(gs[2, 2])

    def _style(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor("#1A1D27")
        for sp in ax.spines.values():
            sp.set_color("#2E3347")
        ax.tick_params(colors="#8892B0", labelsize=8)
        ax.set_title(title, color="white", fontsize=9.5, fontweight="bold", pad=6)
        ax.set_xlabel(xlabel, color="#8892B0", fontsize=8)
        ax.set_ylabel(ylabel, color="#8892B0", fontsize=8)

    # Gate distribution histograms
    for ax, res, col in zip(ax_hist, results, COLOURS):
        ax.hist(res["gate_values"], bins=80, color=col, alpha=0.85, edgecolor="none")
        ax.axvline(CFG["gate_threshold"], color="white", lw=1.2, ls="--", alpha=0.7,
                   label=f"threshold")
        ax.legend(fontsize=7, labelcolor="white", framealpha=0)
        _style(ax, f"Gate Distribution  λ={res['lam']}\n"
                   f"Acc:{res['acc']*100:.1f}%  Sparsity:{res['sparsity']*100:.1f}%",
               "Gate Value", "Count")

    # Training loss curves
    for res, col in zip(results, COLOURS):
        ep = range(1, len(res["history"]["train_loss"]) + 1)
        ax_loss.plot(ep, res["history"]["train_loss"], color=col, lw=2, label=f"λ={res['lam']}")
    ax_loss.legend(fontsize=8, labelcolor="white", framealpha=0)
    _style(ax_loss, "Training Loss", "Epoch", "Total Loss")

    # Test accuracy curves
    for res, col in zip(results, COLOURS):
        ep  = range(1, len(res["history"]["test_acc"]) + 1)
        acc = [a * 100 for a in res["history"]["test_acc"]]
        ax_acc.plot(ep, acc, color=col, lw=2, label=f"λ={res['lam']}")
    ax_acc.legend(fontsize=8, labelcolor="white", framealpha=0)
    _style(ax_acc, "Test Accuracy Over Epochs", "Epoch", "Accuracy (%)")

    # Sparsity growth
    for res, col in zip(results, COLOURS):
        ep = range(1, len(res["history"]["sparsity"]) + 1)
        sp = [s * 100 for s in res["history"]["sparsity"]]
        ax_sp.plot(ep, sp, color=col, lw=2, label=f"λ={res['lam']}")
    ax_sp.legend(fontsize=8, labelcolor="white", framealpha=0)
    _style(ax_sp, "Sparsity Growth", "Epoch", "Sparsity (%)")

    # Lambda vs Accuracy bar chart
    lams = [str(r["lam"]) for r in results]
    accs = [r["acc"] * 100 for r in results]
    bars = ax_bar.bar(lams, accs, color=COLOURS, width=0.5, zorder=3)
    for b, v in zip(bars, accs):
        ax_bar.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%",
                    ha="center", va="bottom", color="white", fontsize=8.5, fontweight="bold")
    ax_bar.set_ylim(0, max(accs) * 1.15)
    ax_bar.grid(axis="y", color="#2E3347", zorder=0)
    _style(ax_bar, "Lambda vs Test Accuracy", "Lambda (λ)", "Accuracy (%)")

    # Accuracy vs Sparsity scatter (key trade-off plot)
    for res, col in zip(results, COLOURS):
        ax_scat.scatter(res["sparsity"] * 100, res["acc"] * 100,
                        color=col, s=130, zorder=5, edgecolors="white", lw=1.5)
        ax_scat.annotate(f"λ={res['lam']}",
                         (res["sparsity"] * 100, res["acc"] * 100),
                         textcoords="offset points", xytext=(7, 4),
                         color=col, fontsize=8, fontweight="bold")
    _style(ax_scat, "Accuracy–Sparsity Trade-off", "Sparsity (%)", "Accuracy (%)")

    # Per-layer sparsity (best lambda model)
    best   = max(results, key=lambda r: r["acc"])
    layers = list(best["per_layer"].keys())
    sps    = list(best["per_layer"].values())
    ys     = range(len(layers))
    ax_layer.barh(list(ys), sps, color=COLOURS[0], alpha=0.85)
    ax_layer.set_yticks(list(ys))
    ax_layer.set_yticklabels(layers, color="#8892B0", fontsize=8)
    ax_layer.set_xlim(0, 100)
    ax_layer.grid(axis="x", color="#2E3347")
    ax_layer.axvline(50, color="white", lw=0.8, ls="--", alpha=0.4)
    for i, v in enumerate(sps):
        ax_layer.text(v + 1, i, f"{v:.1f}%", va="center", color="white", fontsize=8)
    _style(ax_layer, f"Per-Layer Sparsity  (λ={best['lam']})", "Sparsity (%)", "")

    fig.suptitle(
        "Self-Pruning Neural Network — Complete Analysis Dashboard",
        color="white", fontsize=15, fontweight="bold", y=0.96
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  ✓ Dashboard saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────
# 8.  RESULTS TABLE + JSON EXPORT
# ─────────────────────────────────────────────────────────────────────
def print_results_table(results):
    print("\n" + "═" * 60)
    print(f"  {'Lambda':<10}  {'Test Acc':>10}  {'Sparsity':>10}  Notes")
    print("  " + "─" * 56)
    best_acc    = max(r["acc"]      for r in results)
    best_sparse = max(r["sparsity"] for r in results)
    for r in results:
        note = " ← best acc" if r["acc"] == best_acc else (
               " ← most sparse" if r["sparsity"] == best_sparse else "")
        print(f"  {r['lam']:<10}  {r['acc']*100:>9.2f}%  {r['sparsity']*100:>9.2f}%{note}")
    print("═" * 60)


def save_json(results):
    out = [dict(
        lambda_val          = r["lam"],
        test_accuracy_pct   = round(r["acc"] * 100, 2),
        sparsity_pct        = round(r["sparsity"] * 100, 2),
        per_layer_sparsity  = r["per_layer"],
    ) for r in results]
    path = os.path.join(CFG["results_dir"], "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  ✓ JSON saved → {path}")


# ─────────────────────────────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Self-Pruning Neural Network  ·  CIFAR-10               ║")
    print("║   Tredence Analytics — AI Engineering Case Study         ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    train_loader, test_loader = get_loaders()
    results = [run_experiment(lam, train_loader, test_loader) for lam in CFG["lambdas"]]

    print_results_table(results)
    save_json(results)
    make_dashboard(results, os.path.join(CFG["results_dir"], "dashboard.png"))

    print("\nAll done!  ./results/ :")
    print("  dashboard.png  |  results.json")


if __name__ == "__main__":
    main()