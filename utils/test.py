import numpy as np
a = np.arange(63).reshape(-1,7)

b0 =a[0]
b1 =a[1]
b5 = a[5]
b8=a[8]
a[0] = [0,2,3,4,5,2,1]
a[1] = [3,1,3,2,1,2,7]
a[5] = [2,0,7,8,3,1,1]
a[8] = [3,3,1,0,0,2,2]
x_coord = a[:, 0]
y_coord = a[:, 1]
import pdb
pdb.set_trace()
print(a)
box_confidences = a[:,4]
print(box_confidences)
ordered = box_confidences.argsort()[::-1] 
print(ordered)
keep=list()
while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i],
                         x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i],
                         y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)
        iou = intersection / union
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]
