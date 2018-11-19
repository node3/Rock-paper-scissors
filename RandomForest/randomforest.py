import numpy as np
import imageio
from os.path import isfile, join, isdir
from os import listdir, mkdir
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

imageDir = "./augmentedImages"
gestures = ["Rock", "Paper", "Scissors"]
label = {
    "Rock":     0,
    "Paper":    1,
    "Scissors": 2
}
df_y = []
df_x = []

# read images into memory
for gesture in gestures:
    y = label[gesture]
    gesDir = join(imageDir, gesture)
    for f in listdir(gesDir):
        filepath = join(gesDir, f)
        if isfile(filepath) and '.DS_Store' not in filepath:
            image = imageio.imread(filepath, pilmode="L")
            df_x.append(np.reshape(image, (200 * 200)))
            df_y.append(y)
    print("Loaded images of {}".format(gesture))

# check if loaded properly
# plt.imshow(train_x[-1])
# plt.show()

# random forest
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
rf = RandomForestClassifier(n_estimators=500)
rf.fit(x_train, y_train)
print("Classifier generated")

# prediction
count = 0
pred = rf.predict(x_test)
for i in range(len(pred)):
    if pred[i] == y_test[i]:
        count += 1
print(pred)
print(y_test)
print("prediction = {}".format(100*float(count)/float(len(pred))))


