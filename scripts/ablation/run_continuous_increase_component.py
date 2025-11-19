import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
data = {"QNLI dataset": [0.6121, 0.7144, 0.7743, 0.8431]}
base = r"$\mathrm{Base}$"
x = r"$M_x$"
y = r"$M_y$"
z = r"$M_z$"
labels = [
    base,
    f"{base}+{x}",
    f"{base}+{x}+{y}",
    f"{base}+{x}+{y}+{z}"
]
colors = [
    ["
    ["
    ["
    ["
]
target_red = (194/255, 11/255, 11/255)
line_color = '
fig, ax = plt.subplots(1, 1, figsize=(7, 5.5), dpi=160)
dataset_name, values = list(data.items())[0]
values_percent = [v * 100 for v in values]
bar_width = 0.4
gap = 0.25
x_pos = np.arange(len(values_percent)) * (bar_width + gap)
segments = []
for i in range(len(values_percent)):
    if i == 0:
        segments.append([values_percent[0]])
    else:
        segments.append([values_percent[i - 1], values_percent[i] - values_percent[i - 1]])
bars = []
for i, seg in enumerate(segments):
    if i == 0:
        bar = ax.bar(x_pos[i], seg[0],
                     color=colors[i][0],
                     width=bar_width,
                     edgecolor='black',
                     linewidth=1.2)
        bars.append(bar)
    else:
        bottom = 0
        for j, height in enumerate(seg):
            bar = ax.bar(x_pos[i], height,
                         bottom=bottom,
                         color=colors[i][j],
                         width=bar_width,
                         edgecolor='black',
                         linewidth=1.2)
            if j == 0:
                bars.append(bar)
            bottom += height
for value in values_percent:
    ax.axhline(y=value, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
line, = ax.plot(x_pos, values_percent,
                color=line_color,
                linestyle='--',
                linewidth=2,
                marker='o',
                markersize=8,
                markerfacecolor='white',
                markeredgecolor=line_color,
                markeredgewidth=2)
ax.set_xticks([])
ax.set_ylim(60, 88)
ax.set_yticks([60, 65, 70, 75, 80, 85])
ax.tick_params(axis='both', labelsize=11)
offset = 0.5
arr_x_offset = bar_width / 2
text_x_gap = -0.04
end_line_length = 0.15
end_line_offset = -0.2
for i, h in enumerate(values_percent):
    ax.text(x_pos[i], h + offset, f'{h:.2f}%',
            ha='center', va='bottom', fontsize=13)
    if i < len(values_percent) - 1:
        dh = values_percent[i + 1] - values_percent[i]
        arrow_x = x_pos[i] + arr_x_offset
        arrow_y_start = h + offset
        arrow_y_end = values_percent[i + 1] + offset
        arrow = FancyArrowPatch((arrow_x, arrow_y_start),
                                (arrow_x, arrow_y_end),
                                arrowstyle='-|>',
                                mutation_scale=26,
                                color=target_red,
                                lw=3)
        ax.add_patch(arrow)
        end_line_y = arrow_y_end + end_line_offset
        end_line_x_start = arrow_x - end_line_length / 2
        end_line_x_end = arrow_x + end_line_length / 2
        ax.plot([end_line_x_start, end_line_x_end], [end_line_y, end_line_y],
                color=target_red, linewidth=1.5, solid_capstyle='round')
        ax.text(arrow_x + text_x_gap, (arrow_y_start + arrow_y_end) / 2,
                f'{dh:.2f}%',
                ha='right', va='center',
                fontsize=16, color=target_red,
                fontweight='bold')
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor=colors[0][0], edgecolor='black', linewidth=1.2, label=labels[0]),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors[1][0], edgecolor='black', linewidth=1.2, label=labels[1]),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors[2][0], edgecolor='black', linewidth=1.2, label=labels[2]),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors[3][0], edgecolor='black', linewidth=1.2, label=labels[3])
]
line_legend = Line2D(
    [0], [0],
    color=line_color,
    linestyle='--',
    linewidth=2,
    marker='o',
    markersize=8,
    markerfacecolor='white',
    markeredgecolor=line_color,
    markeredgewidth=2,
    label='Accuracy Trend'
)
legend_elements.append(line_legend)
ax.legend(handles=legend_elements,
          loc='upper left',
          fontsize=13,
          framealpha=0.9,
          edgecolor='gray',
          ncol=1)
plt.subplots_adjust(bottom=0.12, top=0.92)
out_path = "qnli_dataset_chart_with_line_legend.png"
fig.savefig(out_path, bbox_inches='tight', dpi=300,
            facecolor='white', edgecolor='none')
print(f"带折线图例的QNLI数据集图表已保存至：{out_path}")
plt.show()