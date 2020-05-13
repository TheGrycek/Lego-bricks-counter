import cv2
import numpy as np
import json
import os

def flash(img_kolor):

    img_hsv = cv2.cvtColor(img_kolor, cv2.COLOR_BGR2HSV)
    threshold = cv2.inRange(img_hsv,  (0, 0, 0), (255, 255, 203))
    threshold = cv2.bitwise_not(threshold)
    threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    blank = np.zeros_like(threshold)
    threshold = img_kolor & threshold
    threshold = cv2.addWeighted(threshold, 0.11, blank, 0.89, 0)
    dst = img_kolor - threshold
    return (dst)

def sharp(img_kolor):

    kernel = np.array([[0, -1, -1], [-1, 11, -1], [-1, -1, -1]], np.float32)
    kernel = 1 / 3 * kernel
    img_c = cv2.filter2D(img_kolor, -1, kernel)
    return(img_c)

def masking(img_kolor):

    kernel = np.array([[0, -1, -1], [-1, 11, -1], [-1, -1, -1]], np.float32)
    kernel = 1 / 3 * kernel
    maska = cv2.filter2D(img_kolor, -1, kernel)
    maska = cv2.blur(maska, (15, 15))
    maska = cv2.Canny(maska, 0, 30, 10)
    maska = cv2.dilate(maska, (20, 20), iterations=20)
    maska = cv2.bitwise_not(maska)
    maska = cv2.blur(maska, (100, 100))
    ret, maska = cv2.threshold(maska, 200, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    maska = cv2.cvtColor(maska, cv2.COLOR_GRAY2BGR)
    return(maska)

def number_conturs(mask_kolor):
    mask = cv2.cvtColor(mask_kolor, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = np.zeros((100, 1), dtype=np.uint16)
    c = int(0)
    for contour in contours:
        area[c] = np.around(cv2.contourArea(contour))
        c = c + 1
    i2 = int(0)
    contours_count_areas = np.zeros_like(contours)
    hierarchy_count_areas = np.zeros(hierarchy.shape[1], dtype=np.uint8)

    for i in range(hierarchy.shape[1]):
        contours_count_areas[i2] = contours[i]
        hierarchy_count_areas[i2] = i2
        i2 = i2 + 1
    return(mask, contours_count_areas, hierarchy_count_areas)

def number_bricks(mask_brick, contours_count_areas, hierarchy_count_areas, color, recognized):
    mask_brick = cv2.dilate(mask_brick, (50, 50), iterations=50)
    contours, hierarchy = cv2.findContours(mask_brick, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    mask_brick2 = np.zeros_like(mask_brick)
    c = int(0)
    area = np.zeros((20000000, 1), dtype=np.uint16)
    for contour in contours:
        area[c] = np.around(cv2.contourArea(contour))
        if area[c] >= 5000:
            cv2.drawContours(mask_brick2, [contour], -1, 255, cv2.FILLED)
        c = c + 1
    mask_brick = mask_brick2

    contours, hierarchy = cv2.findContours(mask_brick, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    i2 = int(0)
    contours_count_brick = np.zeros_like(contours)
    hierarchy_count_brick = np.zeros(hierarchy.shape[1], dtype=np.uint8)
    center = np.zeros([100, 2], dtype=int)

    for i in range(hierarchy.shape[1]):
        M = cv2.moments(contours[i])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        contours_count_brick[i2] = contours[i]
        hierarchy_count_brick[i2] = i2
        center[i2] = [cx, cy]
        i2 = i2 + 1

    for n in range(hierarchy_count_brick.size):

        for cnt in range(contours_count_areas.shape[0]):
            detectorx = 0
            detectory = 0
            for b in contours_count_areas[cnt]:

                    if b[0][0] == center[n][0]:
                        detectorx = detectorx + 1

                    if b[0][1] == center[n][1]:
                        detectory = detectory + 1

                    if detectorx > 0 and detectory > 0:
                        if color == "blue":
                            recognized[cnt, 1] = recognized[cnt, 1] + 1
                            detectorx = 0
                            detectory = 0
                        if color == "red":
                            recognized[cnt, 0] = recognized[cnt, 0] + 1
                            detectorx = 0
                            detectory = 0
                        if color == "yellow":
                            recognized[cnt, 4] = recognized[cnt, 4] + 1
                            detectorx = 0
                            detectory = 0
                        if color == "gray":
                            recognized[cnt, 3] = recognized[cnt, 3] + 1
                            detectorx = 0
                            detectory = 0
                        if color == "white":
                            recognized[cnt, 2] = recognized[cnt, 2] + 1
                            detectorx = 0
                            detectory = 0
    return(recognized)

def assign(read, input):
    c = np.zeros(((read.shape[0]) ** 2, 5))
    k = np.zeros((read.shape[0]) ** 2)
    k_index = np.zeros([(read.shape[0]) ** 2, 2])
    i2 = int(0)
    a_count = int(0)
    b_count = int(0)

    for a in read:
        for b in input:
            for i in range(5):

                if (a[i] == 0 and b[i] != 0) or (b[i] == 0 and a[i] != 0):
                    c[i2][i] = 0

                elif a[i] == 0 and b[i] == 0:
                    c[i2][i] = 0.8

                elif a[i] == b[i]:
                    c[i2][i] = 2

                elif a[i] > b[i]:
                    c[i2][i] = b[i] / a[i]

                elif a[i] < b[i]:
                    c[i2][i] = a[i] / b[i]

            k[i2] = sum(c[i2])
            k_index[i2] = [a_count, b_count]

            i2 = i2 + 1
            b_count = b_count + 1
            if b_count >= input.shape[0]:
                b_count = 0
        a_count = a_count + 1

    k2 = np.zeros((read.shape[0], read.shape[0]))
    k2_index = np.zeros((read.shape[0], read.shape[0], 2))

    num = int(0)

    for ind in range(read.shape[0]):
        for d in range(read.shape[0] ** 2):
            if k_index[d][0] == ind:
                k2[ind][num] = k[d]
                k2_index[ind, num, 0] = k_index[d, 0]
                k2_index[ind, num, 1] = k_index[d, 1]
            num = num + 1
            if num >= read.shape[0]:
                num = 0

    select = np.zeros((read.shape[0], 2), dtype=int)
    select_err = np.zeros((read.shape[0]), dtype=int)

    help = k2
    for p in range(read.shape[0]):
        for ind in range(read.shape[0]):
            git = np.argmax(k2[:, ind])
            select[ind] = k2_index[ind][git]
            select_err[ind] = select[ind][1]
        for i in range(len(select_err)):
            k = i + 1
            for j in range(k, len(select_err)):
                if select_err[i] == select_err[j]:
                    smaller_index = [select[i], select[j]]
                    smaller = [k2[int(smaller_index[0][1]), int(smaller_index[0][0])],
                               k2[int(smaller_index[1][1]), int(smaller_index[1][0])]]
                    a1 = smaller_index[np.argmin(smaller)]
                    help[int(a1[1]), int(a1[0])] = 0
                    git = np.argmax(help[:, int(a1[1])])
                    select[int(a1[1])][1] = git
                    select[ind] = k2_index[ind][git]
    return(select)

def circles_count(img, contours_count_areas, hierarchy_count_areas):

    img_c = sharp(img)
    img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    circles_number = np.zeros([hierarchy_count_areas.size], dtype=int)

    for c in range(hierarchy_count_areas.size):
        mask = np.zeros_like(img_c)
        cv2.drawContours(mask, contours_count_areas, c, 255,  cv2.FILLED)
        mask = img_c & mask
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 70, param1=63, param2=30, minRadius=19, maxRadius=35)

        if hasattr(circles, 'shape'):
            circles_number[c] = circles.shape[1]

    return(circles_number)


def change_type(q):
    if isinstance(q, np.generic) : return q.item()
    raise TypeError

input_path = input("Enter input file path:")
output_path = input("Enter output file path:")
imgs_path = input("Enter images file path:")

def main():

    output = []

    with open(input_path) as json_file:
        data = json.load(json_file)

        imgs_names = []
        for key, value in data.items():
            imgs_names.append(key)

        y = int(1)

        for name in imgs_names:
            img = cv2.imread(f'{imgs_path}\{name}.jpg', 1)
            img_white = img

            input = np.zeros([int(len(data[name])), 5], dtype=int)
            for x in range(len(data[name])):
                input[x] = [data[name][x]['red'], data[name][x]['blue'], data[name][x]['white'], data[name][x]['grey'], data[name][x]['yellow']]

            x, y, ch = img.shape
            img = img[5: x-5, 5: y - 5, :]
            img = img[5: x - 5, 5: y - 5]
            img = masking(img) & img

            mask, contours_count_areas, hierarchy_count_areas = number_conturs(masking(img))
            recognized = np.zeros([hierarchy_count_areas.size, 5], dtype=int)

            circles_number = circles_count(img, contours_count_areas, hierarchy_count_areas)

            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # TODO: MASKI DLA RÓŻNYCH KOLORÓW
            """
            klocki niebieskie
            """
            kernelb = np.ones((7, 7), np.uint8)
            thresholdb = cv2.inRange(img_hsv, (83, 89, 0), (127, 255, 255))
            erode = cv2.erode(thresholdb, kernelb, iterations=1)
            dilationb = cv2.dilate(erode, (2, 2), iterations=1)
            dilationb = cv2.blur(dilationb, (10, 10))
            mask_whiteb = abs(255 -dilationb)

            recognized = number_bricks(dilationb, contours_count_areas, hierarchy_count_areas, "blue", recognized)

            """
            klocki czerwone
            """
            kernel_eroder = np.ones((3, 3), np.uint8)
            kernel_dilater = np.ones((10, 10), np.uint8)
            thresholdr = cv2.inRange(img_hsv, (0, 37, 0), (6, 255, 255))
            eroder = cv2.erode(thresholdr, kernel_eroder, iterations=1)
            dilationr = cv2.dilate(eroder, kernel_dilater, iterations=1)
            dilationr = cv2.blur(dilationr, (20, 20))
            mask_whiter = abs(255 -dilationr)

            recognized = number_bricks(dilationr, contours_count_areas, hierarchy_count_areas, "red", recognized)

            """
            klocki żółte
            """
            kernel_erodey = np.ones((3, 3), np.uint8)
            kernel_dilatey = np.ones((7, 7), np.uint8)
            thresholdy = cv2.inRange(img_hsv, (24, 96, 0), (27, 255, 255))
            erodey = cv2.erode(thresholdy, kernel_erodey, iterations=1)
            dilationy = cv2.dilate(erodey, kernel_dilatey, iterations=1)
            dilationy = cv2.blur(dilationy, (10, 10))
            mask_whitey = abs(255 -dilationy)
            recognized = number_bricks(dilationy, contours_count_areas, hierarchy_count_areas, "yellow", recognized)

            """
            klocki szare
            """
            kernel_erodeg = np.ones((7, 7), np.uint8)
            kernel_dilateg = np.ones((10, 10), np.uint8)
            thresholdg = cv2.inRange(img_hsv, (39, 10, 0), (100, 255, 147))
            erodeg = cv2.erode(thresholdg, kernel_erodeg, iterations=1)
            dilationg = cv2.dilate(erodeg, kernel_dilateg, iterations=1)
            dilationg = cv2.blur(dilationg, (10, 10))
            mask_whiteg = abs(255 - dilationg)
            recognized = number_bricks(dilationg, contours_count_areas, hierarchy_count_areas, "gray", recognized)

            """
            klocki białe
            """
            kernel_erode = np.ones((2, 2), np.uint8)
            img_gray = cv2.cvtColor(img_white, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=500.0, tileGridSize=(20, 20))
            cl1 = clahe.apply(img_gray)
            erode = cv2.erode(cl1, kernel_erode, iterations=8)
            img_bgr = cv2.cvtColor(erode, cv2.COLOR_GRAY2BGR)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            img_all = mask_whiteb & mask_whiteg & mask_whiter & mask_whitey
            img_all = cv2.cvtColor(img_all, cv2.COLOR_GRAY2BGR)
            img_all = cv2.blur(img_all, (8, 8))

            img_all = img_all & img
            img_all2 = img_all

            img_gray = cv2.cvtColor(img_all2, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=500.0, tileGridSize=(20, 20))
            cl1 = clahe.apply(img_gray)
            img_hsv = cv2.cvtColor(img_all, cv2.COLOR_BGR2HSV)
            threshold = cv2.inRange(img_hsv, (0, 0, 0), (33, 255, 255))
            threshold = cv2.dilate(threshold, (5, 5), iterations=8)
            erode = cv2.erode(cl1, (8, 8), iterations=1)

            erode = erode & abs(255 - threshold)
            erode = cv2.blur(erode, (11, 11))
            ret, thresh = cv2.threshold(erode, 160, 255, cv2.THRESH_BINARY)

            white = thresh
            contours, hierarchy = cv2.findContours(white, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            c = int(0)
            mask_white = np.zeros_like(white)
            area = np.zeros((20000000, 1), dtype=np.uint16)
            for contour in contours:
                area[c] = np.around(cv2.contourArea(contour))
                if area[c] >= 5000:
                    cv2.drawContours(mask_white, [contour], -1, 255, cv2.FILLED)
                c = c + 1
            mask_white = cv2.blur(mask_white, (8, 8))

            recognized = number_bricks(mask_white, contours_count_areas, hierarchy_count_areas, "white", recognized)

            """
            wszystkie maski
            """
            read = recognized/2
            select = assign(read, input)
            circles_number_order = np.zeros_like(circles_number)

            for x in range(circles_number.size):
                circles_number_order[x] = circles_number[select[x][1]]

            #print(circles_number_order)
            output.append(list(circles_number_order))

        x = dict(zip(imgs_names, output))

        #print(x)
        with open(output_path, 'w') as q:
            json.dump(x, q, default=change_type)

if __name__ == '__main__':
    main()