"""
Parse training.log and plot train/val loss curves.

Usage:
    python scripts/plot_loss.py
    python scripts/plot_loss.py --log training.log --out plots/loss_curve.png
"""
import argparse
import re
from pathlib import Path


def parse_log(log_path: str) -> tuple[list, list, list, list]:
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    step_pattern = re.compile(r"step\s+(\d+)\s+\|\s+loss\s+([\d.]+)")
    val_pattern = re.compile(r"val_loss\s+([\d.]+)")

    current_step = None
    with open(log_path) as f:
        for line in f:
            m = step_pattern.search(line)
            if m:
                current_step = int(m.group(1))
                train_steps.append(current_step)
                train_losses.append(float(m.group(2)))

            m = val_pattern.search(line)
            if m and current_step is not None:
                val_steps.append(current_step)
                val_losses.append(float(m.group(1)))

    return train_steps, train_losses, val_steps, val_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="training.log")
    parser.add_argument("--out", default="plots/loss_curve.png")
    parser.add_argument("--smooth", type=int, default=50, help="Rolling average window for train loss")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    train_steps, train_losses, val_steps, val_losses = parse_log(args.log)

    if not train_steps:
        print("No training data found in log file.")
        return

    # Rolling average for train loss (raw is noisy)
    def rolling_avg(values, window):
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            result.append(sum(values[start:i+1]) / (i - start + 1))
        return result

    smoothed = rolling_avg(train_losses, args.smooth)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(train_steps, train_losses, color="#d0d0d0", linewidth=0.6, alpha=0.5, label="Train loss (raw)")
    ax.plot(train_steps, smoothed, color="#2563eb", linewidth=2.0, label=f"Train loss (smoothed, window={args.smooth})")

    if val_steps:
        ax.plot(val_steps, val_losses, color="#dc2626", linewidth=2.0,
                marker="o", markersize=4, label="Val loss")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training Loss Curve — GPT on ML ArXiv Papers", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Annotate final values
    if train_losses:
        ax.annotate(f"Final train: {smoothed[-1]:.3f}",
                    xy=(train_steps[-1], smoothed[-1]),
                    xytext=(-80, 15), textcoords="offset points",
                    fontsize=9, color="#2563eb",
                    arrowprops=dict(arrowstyle="->", color="#2563eb"))
    if val_losses:
        ax.annotate(f"Final val: {val_losses[-1]:.3f}",
                    xy=(val_steps[-1], val_losses[-1]),
                    xytext=(-80, -20), textcoords="offset points",
                    fontsize=9, color="#dc2626",
                    arrowprops=dict(arrowstyle="->", color="#dc2626"))

    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    # Print summary stats
    print(f"\nSummary:")
    print(f"  Steps logged:     {len(train_steps)}")
    print(f"  Train loss start: {train_losses[0]:.4f}  →  end: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"  Val loss start:   {val_losses[0]:.4f}  →  end: {val_losses[-1]:.4f}  (best: {min(val_losses):.4f})")

    plt.show()


if __name__ == "__main__":
    main()
