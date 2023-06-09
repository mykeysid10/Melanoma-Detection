import os 
import cv2 

l = ["benign", "malignant"]
for i in range(len(l)):
    for filename in os.listdir(l[i]):
            img = cv2.imread(os.path.join(l[i], filename))[...,::-1]
            img = cv2.resize(img, (300, 300), interpolation = cv2.INTER_AREA)
            cv2.imwrite('{}/{}'.format(l[i], filename), img)