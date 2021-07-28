import matplotlib.pyplot as plt

levels = [1, 2, 3, 4, 5]
base = [37.2987, 37.3711, 36.8420, 36.5719, 37.4241]
hier030303 = [36.9825,37.3935,37.0818,37.1044,37.9705]
hier05025025 = [37.0448,37.4385,37.1205,36.9522,37.7093]
hier060202 = [37.2899,37.3647,36.9425,36.8343,37.6693]
hier080101 = [37.4357,37.4894,37.0134,36.9578,37.6995]

fig, ax1 = plt.subplots()

ax1.set_xlabel('Level')
ax1.set_xticks(levels)
ax1.set_ylabel('mAP')
base = ax1.plot(levels, base, color='tab:gray', linestyle='dashed', label="Baseline")
h1 = ax1.plot(levels, hier030303, color='tab:olive', label="Weights 0.33-0.33-0.33")
h2 = ax1.plot(levels, hier05025025, color='tab:cyan', label="Weights 0.5-0.25-0.25")
h3 = ax1.plot(levels, hier060202, color='tab:brown', label="Weights 0.6-0.2-0.2")
h4 = ax1.plot(levels, hier080101, color='tab:pink', label="Weights 0.8-0.1-0.1")

ax1.tick_params(axis='y')
ax1.grid(True, color='lavender', linewidth=0.5)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels)


fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.savefig('plot_det_weights.png', dpi=300, box_inches='tight')

