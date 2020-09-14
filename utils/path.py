# from pathlib import PureWindowsPath
#
# data_folder = PureWindowsPath("src\\embbeded_dataset\\18.jpg")
#
#
import matplotlib.pyplot as plt
# f = open(data_folder)
#
# print(f.read())
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER,r'gdf', '18.jpg')
img = plt.imread(my_file)
plt.imshow(img)
plt.show()
print(my_file)