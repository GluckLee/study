import os
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib_inline
from PIL import Image

DATADIR = '/Users/v_lijixiang01/workspace/study/LeNet_PM/home/aistudio/work/palm/PALM-Training400/PALM-Training400'
file1 = 'N0012.jpg'
file2 = 'P0095.jpg'

img1 = Image.open(os.path.join(DATADIR, file1))
img1 = np.array(img1)
img2 = Image.open(os.path.join(DATADIR, file2))
img2 = np.array(img2)

plt.figure(figsize=(16, 8))
f = plt.subplot(121)
f.set_title('Normal', fontsize=20)
plt.imshow(img1)
f = plt.subplot(122)
f.set_title('PM', fontsize=20)
plt.imshow(img2)
print(img1.shape, img2.shape)

# plt.show()show
