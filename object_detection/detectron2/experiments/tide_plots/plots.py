import matplotlib.pyplot as plt
import numpy as np

levels = np.arange(1,10)
cls_base = [5.87,6.75,9.12,11.49,9.43,6.00,6.45,7.82,7.57]
loc_base =  [6.85,6.92,6.68,6.31, 6.19,6.57,6.31,6.11,6.03]
fp_base = [19.2,17.96,16.30,14.42,15.18,16.03,14.46,12.65,13.08]
fn_base = [16.16,18.71,21.31,23.64,19.16,13.82,14.09,16.21,15.30]

cls_hier = [5.06,5.77,8.17,10.42,8.39,5.54,5.96,7.29,7.23]
loc_hier = [6.74,6.80,6.77,6.49,6.34,6.70,6.41,6.47,6.33]
fp_hier = [19.82,18.83,16.80,14.94,14.98,15.92,14.24,12.17,12.64]
fn_hier = [14.97,15.98,18.89,21.04,18.69,13.24,13.61,15.86,15.21]


width = 0.4
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey='row')
ax1.bar(levels-width/2, cls_base, width, label="Baseline", color='tab:cyan')
# ax1.plot(levels-width/2, cls_base, linewidth=0.8, color='tab:cyan')
ax1.bar(levels+width/2, cls_hier, width, label="Hierarchical", color='tab:pink')
# ax1.plot(levels+width/2, cls_hier, linewidth=0.8, color='tab:pink')
ax1.set_title("Classification error")
# ax1.set_ylabel("Error")

ax2.bar(levels-width/2, loc_base, width, label="Baseline", color='tab:cyan')
# ax2.plot(levels-width/2, loc_base, linewidth=0.8, color='tab:cyan')
ax2.bar(levels+width/2, loc_hier, width, label="Hierarchical", color='tab:pink')
# ax2.plot(levels+width/2, loc_hier, linewidth=0.8, color='tab:pink')
ax2.set_title("Localization error")
# ax2.set_xlabel("Level")
# ax2.legend()

ax3.bar(levels-width/2, fp_base, width, label="Baseline", color='tab:cyan')
# ax3.plot(levels-width/2, fp_base, linewidth=0.8, color='tab:cyan')
ax3.bar(levels+width/2, fp_hier, width, label="Hierarchical", color='tab:pink')
# ax3.plot(levels+width/2, fp_hier, linewidth=0.8, color='tab:pink')
ax3.set_title("False positives")
#ax3.set_xlabel("Level")

ax4.bar(levels-width/2, fn_base, width, label="Baseline", color='tab:cyan')
# ax4.plot(levels-width/2, fn_base, linewidth=0.8, color='tab:cyan')
ax4.bar(levels+width/2, fn_hier, width, label="Hierarchical", color='tab:pink')
# ax4.plot(levels+width/2, fn_hier, linewidth=0.8, color='tab:pink')
ax4.set_title("False negatives")
fig.legend(handles=ax1.get_legend_handles_labels()[0], labels=ax1.get_legend_handles_labels()[1], loc='center')
fig.tight_layout()
plt.savefig("tide_plots.png", dpi=300)
