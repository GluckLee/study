'''
    # 初次运行时将注释取消，以便解压文件
    # 如果已经解压过，不需要运行此段代码，否则由于文件已经存在，解压时会报错
    unzip -o -q -d /Users/v_lijixiang01/workspace/study/CNN_PM/home/aistudio/work/palm /Users/v_lijixiang01/workspace/study/CNN_PM/home/aistudio/data/data19065/training.zip
    %cd /Users/v_lijixiang01/workspace/study/CNN_PM/home/aistudio/work/palm/PALM-Training400/
    unzip -o -q PALM-Training400.zip
    unzip -o -q -d /Users/v_lijixiang01/workspace/study/CNN_PM/home/aistudio/work/palm /Users/v_lijixiang01/workspace/study/CNN_PM/home/aistudio/data/data19065/validation.zip
    unzip -o -q -d /Users/v_lijixiang01/workspace/study/CNN_PM/home/aistudio/work/palm /Users/v_lijixiang01/workspace/study/CNN_PM/home/aistudio/data/data19065/valid_gt.zip
    #返回家目录，生成模型文件位于/home/aistudio/
    %cd /home/aistudio/
'''

import os
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib_inline
from PIL import Image

DATADIR = '/Users/v_lijixiang01/workspace/study/CNN_PM/home/aistudio/work/palm/PALM-Training400/PALM-Training400'
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

# plt.show()
