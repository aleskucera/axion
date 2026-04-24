from pathlib import Path
import csv
import textwrap
import matplotlib.pyplot as plt

CSV_PATH = Path(__file__).with_name("contact_mtl_lambda_regr_only.csv")


def format_value(value: float) -> str:
    """Use scientific notation for large values."""
    if abs(value) > 1000:
        return f"{value:.2e}"
    return f"{value:.2f}"

def add_bar_labels(ax: plt.Axes, bars) -> None:
    ymax = ax.get_ylim()[1]
    offset = 0.01 * ymax if ymax else 0.0
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            format_value(height),
            ha="center",
            va="bottom",
        )

def wrap_model_names(model_names: list[str], line_width: int = 16) -> list[str]:
    wrapped_names = []
    for i, name in enumerate(model_names, start=1):
        compact = " ".join(name.split())
        wrapped = textwrap.fill(compact, width=line_width, break_long_words=False)
        wrapped_names.append(f"{i}: {wrapped}")
    return wrapped_names


def style_x_labels(ax: plt.Axes) -> None:
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("center")


def plot_metric(model_names, values, title: str, y_label: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(model_names, values, color=color)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    style_x_labels(ax)
    add_bar_labels(ax, bars)
    fig.tight_layout()


def add_grouped_bar_labels(ax: plt.Axes, bars) -> None:
    ymax = ax.get_ylim()[1]
    offset = 0.01 * ymax if ymax else 0.0
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            format_value(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_state_lambda_mae(
    model_names: list[str],
    state_maes: list[float],
    lambda_maes: list[float],
) -> None:
    n = len(model_names)
    x = list(range(n))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_state = ax.bar(
        [i - width / 2 for i in x],
        state_maes,
        width,
        label="State MAE",
        color="tab:green",
    )
    bars_lambda = ax.bar(
        [i + width / 2 for i in x],
        lambda_maes,
        width,
        label="Lambda MAE",
        color="tab:red",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_title("State MAE vs Lambda MAE by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Absolute Error")
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    style_x_labels(ax)
    add_grouped_bar_labels(ax, bars_state)
    add_grouped_bar_labels(ax, bars_lambda)
    fig.tight_layout()


def main() -> None:
    with CSV_PATH.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No data found in {CSV_PATH}")

    model_names = [
        (row.get("model_info") or f"model_{i + 1}").strip()
        for i, row in enumerate(rows)
    ]
    display_names = wrap_model_names(model_names)
    total_abs_error = [float(row["total_absolute_error"]) for row in rows]
    lambda_total_abs_error = [float(row["lambda_total_absolute_error"]) for row in rows]
    total_sq_error = [float(row["total_squared_error"]) for row in rows]
    state_maes = [float(row["state_mae"]) for row in rows]
    lambda_maes = [float(row["lambda_mae"]) for row in rows]

    plot_metric(
        model_names=display_names,
        values=total_abs_error,
        title="Total Absolute Error by Model",
        y_label="Total Absolute Error",
        color="tab:blue",
    )
    plot_metric(
        model_names=display_names,
        values=lambda_total_abs_error,
        title="Total Absolute Lambda Error by Model",
        y_label="Total Absolute Lambda Error",
        color="tab:purple",
    )
    plot_metric(
        model_names=display_names,
        values=total_sq_error,
        title="Total Squared Error by Model",
        y_label="Total Squared Error",
        color="tab:orange",
    )
    plot_state_lambda_mae(display_names, state_maes, lambda_maes)

    print("\nModel label mapping:")
    for i, name in enumerate(model_names, start=1):
        print(f"{i}: {name}")

    plt.show()


if __name__ == "__main__":
    main()
