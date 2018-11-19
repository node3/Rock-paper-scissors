import numpy as np
import pandas as pd
import imageio
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
# from sklearn.cross_validation import train_test_split

data = imageio.imread("augmentedImages/Paper/1.jpg", pilmode="RGB")
plt.imshow(data)
plt.show()