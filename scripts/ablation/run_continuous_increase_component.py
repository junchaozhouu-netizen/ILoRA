import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


def main() -> None:
    data = {"QNLI dataset": [0.6121, 0.7144, 0.7743, 0.8431]}

    base = r"$\mathrm{Base}$"
    x_token = r"$M_x$"
    y_token = r"$M_y$"
    z_token = r"$M_z$"

    labels = [
        base,
        f"{base}+{x_token}",
        f"{base}+{x_token}+{y_token}",
        f"{base}+{x_token}+{y_token}+{z_token}",
    ]

    colors = [
        ["#d9d9d9"],
        ["#bdbdbd", "#9ecae1"],
        ["#969696", "#9ecae1"],
        ["#737373", "#9ecae1"],
    ]
    arrow_color = (194 / 255, 11 / 255, 11 / 255)
    line_color = "#2b6cb0"

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=160)
    _, values = list(data.items())[0]
    values_percent = [value * 100 for value in values]

    bar_width = 0.4
    gap = 0.25
    x_pos = np.arange(len(values_percent)) * (bar_width + gap)

    segments = []
    for idx, value in enumerate(values_percent):
        if idx == 0:
            segments.append([value])
        else:
            segments.append([values_percent[idx - 1], value - values_percent[idx - 1]])

    for idx, segment in enumerate(segments):
        if idx == 0:
            ax.bar(
                x_pos[idx],
                segment[0],
                color=colors[idx][0],
                width=bar_width,
                edgecolor="black",
                linewidth=1.2,
            )
            continue

        bottom = 0
        for seg_idx, height in enumerate(segment):
            ax.bar(
                x_pos[idx],
                height,
                bottom=bottom,
                color=colors[idx][seg_idx],
                width=bar_width,
                edgecolor="black",
                linewidth=1.2,
            )
            bottom += height

    for value in values_percent:
        ax.axhline(y=value, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.plot(
        x_pos,
        values_percent,
        color=line_color,
        linestyle="--",
        linewidth=2,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor=line_color,
        markeredgewidth=2,
    )

    ax.set_xticks([])
    ax.set_ylim(60, 88)
    ax.set_yticks([60, 65, 70, 75, 80, 85])
    ax.tick_params(axis="both", labelsize=11)

    offset = 0.5
    arrow_x_offset = bar_width / 2
    text_x_gap = -0.04
    end_line_length = 0.15
    end_line_offset = -0.2

    for idx, height in enumerate(values_percent):
        ax.text(x_pos[idx], height + offset, f"{height:.2f}%", ha="center", va="bottom", fontsize=13)
        if idx >= len(values_percent) - 1:
            continue

        delta = values_percent[idx + 1] - values_percent[idx]
        arrow_x = x_pos[idx] + arrow_x_offset
        arrow_y_start = height + offset
        arrow_y_end = values_percent[idx + 1] + offset

        arrow = FancyArrowPatch(
            (arrow_x, arrow_y_start),
            (arrow_x, arrow_y_end),
            arrowstyle="-|>",
            mutation_scale=26,
            color=arrow_color,
            lw=3,
        )
        ax.add_patch(arrow)

        end_line_y = arrow_y_end + end_line_offset
        end_line_x_start = arrow_x - end_line_length / 2
        end_line_x_end = arrow_x + end_line_length / 2
        ax.plot(
            [end_line_x_start, end_line_x_end],
            [end_line_y, end_line_y],
            color=arrow_color,
            linewidth=1.5,
            solid_capstyle="round",
        )
        ax.text(
            arrow_x + text_x_gap,
            (arrow_y_start + arrow_y_end) / 2,
            f"{delta:.2f}%",
            ha="right",
            va="center",
            fontsize=16,
            color=arrow_color,
            fontweight="bold",
        )

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[0][0], edgecolor="black", linewidth=1.2, label=labels[0]),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[1][0], edgecolor="black", linewidth=1.2, label=labels[1]),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[2][0], edgecolor="black", linewidth=1.2, label=labels[2]),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[3][0], edgecolor="black", linewidth=1.2, label=labels[3]),
        Line2D(
            [0],
            [0],
            color=line_color,
            linestyle="--",
            linewidth=2,
            marker="o",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor=line_color,
            markeredgewidth=2,
            label="Accuracy trend",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=13,
        framealpha=0.9,
        edgecolor="gray",
        ncol=1,
    )

    plt.subplots_adjust(bottom=0.12, top=0.92)
    out_path = "qnli_dataset_chart_with_line_legend.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none")
    print(f"Saved chart to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
