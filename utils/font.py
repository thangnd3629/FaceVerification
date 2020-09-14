from os.path import isdir
from PIL import Image
import os
import matplotlib.pyplot as plt
#

os.chdir(r"C:")
print(os.getcwd())
img = plt.imread(r"C:\Users\SeniorDeV\Desktop\117618984_341678670570352_5637829890153900274_n.jpg")
img = plt.imread(r"F:\thangtest.jpg")
l = plt.imread("‪D:\FaceVerification\cw2_frames\Huy\\18.jpg")
plt.imshow(img)

plt.show()