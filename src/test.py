import os
import matplotlib.pyplot as plt
print(os.getcwd())
parent = os.path.dirname(os.getcwd())
os.chdir(parent)
print(os.getcwd())
#same folder ok
x = plt.imread(r"C:\Program Files\UniKey\Capture.PNG")

plt.imshow(x)
plt.show()
print(os.getcwd())