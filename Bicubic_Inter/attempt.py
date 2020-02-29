import cv2
from sklearn.metrics import mean_squared_error
# comparison between given example_hr and interpolated example_hr

img_given = cv2.imread('img_example_hr.png')
H,W,C = img_given.shape
img_given = img_given[0:H - 1, 0:W - 1]
print(img_given.shape)
cv2.imwrite('img_example_hr.png', img_given)


img_inter = cv2.imread('img_example_lr_test.png')
print(img_inter.shape)
