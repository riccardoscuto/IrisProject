import cv2 as cv
import numpy as np

def draw_circles(storage, output):
    circles = np.squeeze(storage)
    for circle in circles:
        radius, x, y = int(circle[2]), int(circle[0]), int(circle[1])
        cv.circle(output, (x, y), 1, (0, 255, 0), -1, cv.LINE_8, 0)
        cv.circle(output, (x, y), radius, (255, 0, 0), 3, cv.LINE_8, 0)

orig = cv.imread('images/R/S1001R02.jpg')
processed = cv.imread('images/R/S1001R02.jpg', cv.IMREAD_GRAYSCALE)
circles = cv.HoughCircles(processed, cv.HOUGH_GRADIENT, dp=2, minDist=100, param1=100, param2=30, minRadius=60, maxRadius=150)
if circles is not None:
    circles = np.uint16(np.around(circles))
    draw_circles(circles[0], orig)

cv.namedWindow("original with circles", cv.WINDOW_NORMAL)
cv.imshow("original with circles", orig)
cv.waitKey(0)
cv.destroyAllWindows()
