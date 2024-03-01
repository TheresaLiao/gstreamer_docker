import cv2
import numpy as np

print(np.pi)
img = cv2.imread('./test.jpg')

cv2.imshow('imagen-zeros',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
