import matplotlib.pyplot as plt
from PIL import Image

fig, axes = plt.subplots(2, 2)


axes[0,0].imshow(Image.open('concat1_part1.png'), aspect = 'equal')
axes[0,1].imshow(Image.open('concat1_part2.png'), aspect = 'equal')
# axes[0,2].imshow(Image.open('concat1_part3.png'), aspect='auto')
# axes[1,0].imshow(Image.open('concat2_part1.png'), aspect ='equal')
# axes[1,1].imshow(Image.open('concat2_part2.png'), aspect = 'equal')
# axes[1,2].imshow(Image.open('concat2_part3.png'), aspect='auto')
axes[1,0].imshow(Image.open('concat3_part1.png'), aspect='equal')
axes[1,1].imshow(Image.open('concat3_part2.png'), aspect='equal')
# axes[2,2].imshow(Image.open('concat3_part3.png'))

for i in range(2):
    for j in range(2):
        axes[i,j].set_axis_off()

plt.axis('off')
plt.subplots_adjust(wspace=0.0, hspace=0.0)
fig.tight_layout()
plt.savefig('figure1_2.png', bbox_inches='tight', dpi=300)
