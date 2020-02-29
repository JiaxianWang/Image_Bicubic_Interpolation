import cv2
import numpy as np
import math
# runtime
from datetime import datetime

start = datetime.now()

org_img = 'img_example_lr.png'

img_read = cv2.imread(org_img)

print(img_read.shape)

H, W, C = img_read.shape

img_new = np.zeros((2 * H, 2 * W, C))

## PADDING
for h in range(H):
    for w in range(W):
        for c in range(C):
            img_new[h * 2, w * 2, c] = img_read[h, w, c]

# check
for h in range(H):
    for w in range(W):
        for c in range(C):
            if img_new[h * 2, w * 2, c] != img_read[h, w, c]:
                print("problem")
            else:
                continue
print("ok")

# interpolation for shaded dots (center missing dots)
for i in range(3, 2 * H - 3, 2):
    for j in range(3, 2 * W - 3, 2):
        for c in range(C):
            matF = np.matrix([[img_new[int(i - 3), int(j - 3), c], img_new[int(i - 3), int(j - 1), c],
                               img_new[int(i - 3), int(j + 1), c], img_new[int(i - 3), int(j + 3), c]],
                              [img_new[int(i - 1), int(j - 3), c], img_new[int(i - 1), int(j - 1), c],
                               img_new[int(i - 1), int(j + 1), c], img_new[int(i - 1), int(j + 3), c]],
                              [img_new[int(i + 1), int(j - 3), c], img_new[int(i + 1), int(j - 1), c],
                               img_new[int(i + 1), int(j + 1), c], img_new[int(i + 1), int(j + 3), c]],
                              [img_new[int(i + 3), int(j - 3), c], img_new[int(i + 3), int(j - 1), c],
                               img_new[int(i + 3), int(j + 1), c], img_new[int(i + 3), int(j + 3), c]]])
            coef_h = np.matrix([-1 / 16, 9 / 16, 9 / 16, -1 / 16])
            coef_l = np.matrix([[-1 / 16], [9 / 16], [9 / 16], [-1 / 16]])
            img_new[i, j, c] = np.dot(coef_h, np.dot(matF, coef_l))

# row & column interpolation for missing dots
# do column interpolation first
for j in range(4, 2 * W - 3, 2):
    for i in range(3, 2 * H - 3, 2):
        for c in range(C):
            col_F = np.matrix(
                [img_new[int(i - 3), int(j), c], img_new[int(i - 1), int(j), c], img_new[int(i + 1), int(j), c],
                 img_new[int(i + 3), int(j), c]])
            coef_F = np.matrix([[-1 / 16], [9 / 16], [9 / 16], [-1 / 16]])

            img_new[i, j, c] = np.dot(col_F, coef_F)

# do row interpolation for missing dots
for i in range(4, 2 * H - 3, 2):
    for j in range(3, 2 * W - 3, 2):
        for c in range(C):
            row_F = np.matrix(
                [[img_new[int(i), int(j - 3), c]], [img_new[int(i), int(j - 1), c]], [img_new[int(i), int(j + 1), c]],
                 [img_new[int(i), int(j + 3), c]]])
            coef_R = np.matrix([-1 / 16, 9 / 16, 9 / 16, -1 / 16])

            img_new[i, j, c] = np.dot(coef_R, row_F)


# interpolation for edges (missing dots)
# condition function
def con_add(a, b):
    result = a + b
    if result > 2 * W - 1:
        result = a + b - 2 * (a + b - (2 * W - 1))
    return result


def con_sub(a, b):
    result = a - b
    if result < 0:
        result = a - b + 2
    return result


# row 0 to 2 and H-3 to H-1
# 0 to 2 (row 0 and row 2)
for i in range(0, 3, 2):
    for j in range(1, 2 * W, 2):
        for c in range(C):
            row_F = np.matrix(
                [[img_new[int(i), int(con_sub(j, 3)), c]], [img_new[int(i), int(con_sub(j, 1)), c]],
                 [img_new[int(i), int(con_add(j, 1)), c]],
                 [img_new[int(i), int(con_add(j, 3)), c]]])
            coef_R = np.matrix([-1 / 16, 9 / 16, 9 / 16, -1 / 16])

            img_new[i, j, c] = np.dot(coef_R, row_F)

# H-3 to H-1(row 598 only)
for j in range(1, 2 * W, 2):
    for c in range(C):
        row_F = np.matrix(
            [[img_new[int(2 * H - 2), int(con_sub(j, 3)), c]], [img_new[int(2 * H - 2), int(con_sub(j, 1)), c]],
             [img_new[int(2 * H - 2), int(con_add(j, 1)), c]],
             [img_new[int(2 * H - 2), int(con_add(j, 3)), c]]])
        coef_R = np.matrix([-1 / 16, 9 / 16, 9 / 16, -1 / 16])

        img_new[2 * H - 2, j, c] = np.dot(coef_R, row_F)

# row 1: center missing dots first
for j in range(1, 2 * W, 2):
    for c in range(C):
        matF = np.matrix(
            [[img_new[int(con_sub(1, 3)), int(con_sub(j, 3)), c], img_new[int(con_sub(1, 3)), int(con_sub(j, 1)), c],
              img_new[int(con_sub(1, 3)), int(con_add(j, 1)), c], img_new[int(con_sub(1, 3)), int(con_add(j, 3)), c]],
             [img_new[int(con_sub(1, 1)), int(con_sub(j, 3)), c], img_new[int(con_sub(1, 1)), int(con_sub(j, 1)), c],
              img_new[int(con_sub(1, 1)), int(con_add(j, 1)), c], img_new[int(con_sub(1, 1)), int(con_add(j, 3)), c]],
             [img_new[int(con_add(1, 1)), int(con_sub(j, 3)), c], img_new[int(con_add(1, 1)), int(con_sub(j, 1)), c],
              img_new[int(con_add(1, 1)), int(con_add(j, 1)), c], img_new[int(con_add(1, 1)), int(con_add(j, 3)), c]],
             [img_new[int(con_add(1, 3)), int(con_sub(j, 3)), c], img_new[int(con_add(1, 3)), int(con_sub(j, 1)), c],
              img_new[int(con_add(1, 3)), int(con_add(j, 1)), c], img_new[int(con_add(1, 3)), int(con_add(j, 3)), c]]])
        coef_h = np.matrix([-1 / 16, 9 / 16, 9 / 16, -1 / 16])
        coef_l = np.matrix([[-1 / 16], [9 / 16], [9 / 16], [-1 / 16]])
        img_new[1, j, c] = np.dot(coef_h, np.dot(matF, coef_l))

# row 1: white missing dots
for j in range(0, 2 * W - 1, 2):
    for c in range(C):
        col_F = np.matrix(
            [img_new[int(con_sub(1, 3)), int(j), c], img_new[int(1 - 1), int(j), c], img_new[int(1 + 1), int(j), c],
             img_new[int(1 + 3), int(j), c]])
        coef_F = np.matrix([[-1 / 16], [9 / 16], [9 / 16], [-1 / 16]])

        img_new[1, j, c] = np.dot(col_F, coef_F)


# row 597 center missing dots first
def add_599(a, b):
    result = a + b
    if result > 2 * H - 1:
        result = a + b - 2 * (a + b - (2 * H - 1))
    return result


for j in range(1, 2 * W, 2):
    for c in range(C):
        matF = np.matrix(
            [[img_new[int(con_sub(2 * H - 3, 3)), int(con_sub(j, 3)), c],
              img_new[int(con_sub(2 * H - 3, 3)), int(con_sub(j, 1)), c],
              img_new[int(con_sub(2 * H - 3, 3)), int(add_599(j, 1)), c],
              img_new[int(con_sub(2 * H - 3, 3)), int(add_599(j, 3)), c]],
             [img_new[int(con_sub(2 * H - 3, 1)), int(con_sub(j, 3)), c],
              img_new[int(con_sub(2 * H - 3, 1)), int(con_sub(j, 1)), c],
              img_new[int(con_sub(2 * H - 3, 1)), int(add_599(j, 1)), c],
              img_new[int(con_sub(2 * H - 3, 1)), int(add_599(j, 3)), c]],
             [img_new[int(add_599(2 * H - 3, 1)), int(con_sub(j, 3)), c],
              img_new[int(add_599(2 * H - 3, 1)), int(con_sub(j, 1)), c],
              img_new[int(add_599(2 * H - 3, 1)), int(add_599(j, 1)), c],
              img_new[int(add_599(2 * H - 3, 1)), int(add_599(j, 3)), c]],
             [img_new[int(add_599(2 * H - 3, 3)), int(con_sub(j, 3)), c],
              img_new[int(add_599(2 * H - 3, 3)), int(con_sub(j, 1)), c],
              img_new[int(add_599(2 * H - 3, 3)), int(add_599(j, 1)), c],
              img_new[int(add_599(2 * H - 3, 3)), int(add_599(j, 3)), c]]])
        coef_h = np.matrix([-1 / 16, 9 / 16, 9 / 16, -1 / 16])
        coef_l = np.matrix([[-1 / 16], [9 / 16], [9 / 16], [-1 / 16]])
        img_new[2 * H - 3, j, c] = np.dot(coef_h, np.dot(matF, coef_l))

# row 597: white missing dots
for j in range(0, 2 * W - 1, 2):
    for c in range(C):
        col_F = np.matrix(
            [img_new[int(con_sub(2 * H - 3, 3)), int(j), c], img_new[int(2 * H - 3 - 1), int(j), c],
             img_new[int(2 * H - 3 + 1), int(j), c],
             img_new[int(add_599(2 * H - 3, 3)), int(j), c]])
        coef_F = np.matrix([[-1 / 16], [9 / 16], [9 / 16], [-1 / 16]])

        img_new[2 * H - 3, j, c] = np.dot(col_F, coef_F)

# no data for row = 599

# column 0 to 2 and H-3 to H-1
# column interpolation for column 0 and 2
# column 0
for i in range(3, 2 * H - 4, 2):
    for c in range(C):
        col_F = np.matrix(
            [img_new[int(i - 3), int(0), c], img_new[int(i - 1), int(0), c], img_new[int(i + 1), int(0), c],
             img_new[int(i + 3), int(0), c]])
        coef_F = np.matrix([[-1 / 16], [9 / 16], [9 / 16], [-1 / 16]])

        img_new[i, 0, c] = np.dot(col_F, coef_F)
# column 2
for i in range(3, 2 * H - 4, 2):
    for c in range(C):
        col_F = np.matrix(
            [img_new[int(i - 3), int(2), c], img_new[int(i - 1), int(2), c], img_new[int(i + 1), int(2), c],
             img_new[int(i + 3), int(2), c]])
        coef_F = np.matrix([[-1 / 16], [9 / 16], [9 / 16], [-1 / 16]])

        img_new[i, 2, c] = np.dot(col_F, coef_F)
# center missing white dots in column 1
for i in range(3, 2 * H - 4, 2):
    for c in range(C):
        matF = np.matrix([[img_new[int(i - 3), int(con_sub(1, 3)), c], img_new[int(i - 3), int(1 - 1), c],
                           img_new[int(i - 3), int(1 + 1), c], img_new[int(i - 3), int(1 + 3), c]],
                          [img_new[int(i - 1), int(con_sub(1, 3)), c], img_new[int(i - 1), int(1 - 1), c],
                           img_new[int(i - 1), int(1 + 1), c], img_new[int(i - 1), int(1 + 3), c]],
                          [img_new[int(i + 1), int(con_sub(1, 3)), c], img_new[int(i + 1), int(1 - 1), c],
                           img_new[int(i + 1), int(1 + 1), c], img_new[int(i + 1), int(1 + 3), c]],
                          [img_new[int(i + 3), int(con_sub(1, 3)), c], img_new[int(i + 3), int(1 - 1), c],
                           img_new[int(i + 3), int(1 + 1), c], img_new[int(i + 3), int(1 + 3), c]]])
        coef_h = np.matrix([-1 / 16, 9 / 16, 9 / 16, -1 / 16])
        coef_l = np.matrix([[-1 / 16], [9 / 16], [9 / 16], [-1 / 16]])

        img_new[i, 1, c] = np.dot(coef_h, np.dot(matF, coef_l))

# other missing dots in column 1
for i in range(4, 2 * H - 3, 2):
    for c in range(C):
        row_F = np.matrix(
            [[img_new[int(i), int(con_sub(1, 3)), c]], [img_new[int(i), int(1 - 1), c]],
             [img_new[int(i), int(1 + 1), c]],
             [img_new[int(i), int(1 + 3), c]]])
        coef_R = np.matrix([-1 / 16, 9 / 16, 9 / 16, -1 / 16])

        img_new[i, 1, c] = np.dot(coef_R, row_F)
# column 797 and 798 left
# column 798: column interpolation
for i in range(3, 2 * H - 4, 2):
    for c in range(C):
        col_F = np.matrix(
            [img_new[int(i - 3), int(2 * W - 2), c], img_new[int(i - 1), int(2 * W - 2), c],
             img_new[int(i + 1), int(2 * W - 2), c],
             img_new[int(i + 3), int(2 * W - 2), c]])
        coef_F = np.matrix([[-1 / 16], [9 / 16], [9 / 16], [-1 / 16]])

        img_new[i, 2 * W - 2, c] = np.dot(col_F, coef_F)

# center missing white dots in column 797
for i in range(3, 2 * H - 4, 2):
    for c in range(C):
        matF = np.matrix(
            [[img_new[int(con_sub(i, 3)), int(con_sub(2 * W - 3, 3)), c],
              img_new[int(con_sub(i, 3)), int(con_sub(2 * W - 3, 1)), c],
              img_new[int(con_sub(i, 3)), int(con_add(2 * W - 3, 1)), c],
              img_new[int(con_sub(i, 3)), int(con_add(2 * W - 3, 3)), c]],
             [img_new[int(con_sub(i, 1)), int(con_sub(2 * W - 3, 3)), c],
              img_new[int(con_sub(i, 1)), int(con_sub(2 * W - 3, 1)), c],
              img_new[int(con_sub(i, 1)), int(con_add(2 * W - 3, 1)), c],
              img_new[int(con_sub(i, 1)), int(con_add(2 * W - 3, 3)), c]],
             [img_new[int(con_add(i, 1)), int(con_sub(2 * W - 3, 3)), c],
              img_new[int(con_add(i, 1)), int(con_sub(2 * W - 3, 1)), c],
              img_new[int(con_add(i, 1)), int(con_add(2 * W - 3, 1)), c],
              img_new[int(con_add(i, 1)), int(con_add(2 * W - 3, 3)), c]],
             [img_new[int(con_add(i, 3)), int(con_sub(2 * W - 3, 3)), c],
              img_new[int(con_add(i, 3)), int(con_sub(2 * W - 3, 1)), c],
              img_new[int(con_add(i, 3)), int(con_add(2 * W - 3, 1)), c],
              img_new[int(con_add(i, 3)), int(con_add(2 * W - 3, 3)), c]]])
        coef_h = np.matrix([-1 / 16, 9 / 16, 9 / 16, -1 / 16])
        coef_l = np.matrix([[-1 / 16], [9 / 16], [9 / 16], [-1 / 16]])

        img_new[i, 2 * W - 3, c] = np.dot(coef_h, np.dot(matF, coef_l))
# the other missing dots using row interpolation
for i in range(4, 2 * H - 3, 2):
    for c in range(C):
        row_F = np.matrix(
            [[img_new[int(i), int(2 * W - 3 - 3), c]], [img_new[int(i), int(2 * W - 3 - 1), c]],
             [img_new[int(i), int(2 * W - 3 + 1), c]],
             [img_new[int(i), int(con_add(2 * W - 3, 3)), c]]])
        coef_R = np.matrix([-1 / 16, 9 / 16, 9 / 16, -1 / 16])

        img_new[i, 2 * W - 3, c] = np.dot(coef_R, row_F)

# crop the last column and last row
img_new = img_new[0:2 * H - 1, 0:2 * W - 1]
print(img_new.shape)

cv2.imwrite('img_example_lr_test.png', img_new)

print(datetime.now() - start)
