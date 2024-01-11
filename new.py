import cv2 as cv
import numpy as np
import os

centroid = (0, 0)
radius = 0
current_eye = 0
eyes_list = []

def get_new_eye(lst):
    global current_eye
    if current_eye >= len(lst):
        current_eye = 0
    new_eye = lst[current_eye]
    current_eye += 1
    return new_eye

def get_iris(frame):
    iris = []
    copy_img = frame.copy()
    res_img = frame.copy()
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_img)
    edges = cv.Canny(gray_img, 5, 70, 3)
    gray_img = cv.GaussianBlur(gray_img, (7, 7), 0)
    circles = get_circles(edges)
    iris.append(res_img)
    for circle in circles:
        rad = int(circle[0][2])
        global radius
        radius = rad
        cv.circle(mask, centroid, rad, (255, 255, 255), cv.FILLED)
        mask = cv.bitwise_not(mask)
        cv.subtract(frame, copy_img, res_img, mask)
        x = int(centroid[0] - rad)
        y = int(centroid[1] - rad)
        w = int(rad * 2)
        h = w
        crop_img = res_img[y:y + h, x:x + w].copy()
        return crop_img
    return res_img

def get_circles(image):
    i = 80
    while i < 151:
        circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 2, 100.0,
                                  param1=30, param2=i, minRadius=100, maxRadius=140)
        if circles is not None and len(circles) == 1:
            return circles
        i += 1
    return []

def get_pupil(frame):
    pupil_img = np.zeros_like(frame)
    lower_bound = np.array([30, 30, 30])
    upper_bound = np.array([80, 80, 80])
    mask = cv.inRange(frame, lower_bound, upper_bound)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    pupil_img = frame.copy()
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 50:
            moments = cv.moments(contour)
            x = int(moments['m10'] / moments['m00'])
            y = int(moments['m01'] / moments['m00'])
            global centroid
            centroid = (x, y)
            cv.drawContours(pupil_img, [contour], -1, (0, 0, 0), cv.FILLED)
            break
    return pupil_img

def get_polar_to_cart_img(image, rad):
    c = (float(image.shape[1] / 2.0), float(image.shape[0] / 2.0))
    img_res = cv.logPolar(image, c, rad, cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS)
    return img_res


output_folder = 'results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cv.namedWindow("input", cv.WINDOW_GUI_NORMAL)
cv.namedWindow("output", cv.WINDOW_NORMAL)
cv.namedWindow("normalized", cv.WINDOW_GUI_NORMAL)

cap = cv.VideoCapture(0)  # Inizializzazione della webcam

# Il codice precedente rimane invariato fino a questo punto

while True:
    ret, frame = cap.read()  # Acquisizione del frame dalla webcam
    if not ret:
        print("Impossibile leggere il frame dalla webcam.")
        break

    iris = frame.copy()
    output = get_pupil(frame)
    iris = get_iris(output)

    # Verifica se l'immagine dell'iride è valida prima di eseguire la conversione
    if not iris.size:
        continue

    # Trova il cerchio dell'iride
    circles = get_circles(cv.Canny(cv.cvtColor(iris, cv.COLOR_BGR2GRAY), 5, 70, 3))
    if circles is not None and len(circles) > 0:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Zoom sull'iride
            x, y, r = i[0], i[1], i[2]
            zoomed_iris = iris[max(0, y - r):y + r, max(0, x - r):x + r]

            cv.imshow("input", frame)

            # Disegna il cerchio solo se l'iride è stata trovata
            cv.circle(iris, (i[0], i[1]), i[2], (0, 0, 255), 2)
            cv.imshow("output", iris)

            if zoomed_iris.size > 0:
                cv.imshow("normalized", get_polar_to_cart_img(zoomed_iris, radius))

    key = cv.waitKey(1)
    if key == 27 or key == 1048603:
        # Salvataggio dei risultati
        cv.imwrite(os.path.join(output_folder, f"frame_input_{current_eye}.png"), frame)
        cv.imwrite(os.path.join(output_folder, f"frame_output_{current_eye}.png"), iris)
        cv.imwrite(os.path.join(output_folder, f"frame_normalized_{current_eye}.png"), get_polar_to_cart_img(iris, radius))
        
        current_eye += 1
        if current_eye >= len(eyes_list):
            break

# Il resto del codice rimane invariato


cap.release()
cv.destroyAllWindows()