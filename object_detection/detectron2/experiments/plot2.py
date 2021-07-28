import matplotlib.pyplot as plt

levels = [1, 2, 3, 4, 5, 6]
levels4 = [37.0241,37.2134,36.8262,36.8374,37.5420,40.5817]
levels5 = [37.1580,37.2553,36.8221,36.7949,37.7424,41.1339]
levels6 = [36.9602,37.3270,36.9767,37.1438,38.1778,41.3280]


fig, ax1 = plt.subplots()

ax1.set_xlabel('Level')
ax1.set_xticks(levels)
ax1.set_ylabel('mAP')
l4 = ax1.plot(levels, levels4, color='tab:olive', label="4 levels")
l5 = ax1.plot(levels, levels5, color='tab:cyan', label="5 levels")
l6 = ax1.plot(levels, levels6, color='tab:pink', label="6 levels")

ax1.tick_params(axis='y')
ax1.grid(True, color='lavender', linewidth=0.5)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels)


fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.savefig('plot_det_levels.png', dpi=300, box_inches='tight')

