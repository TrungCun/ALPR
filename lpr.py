
import cv2
import re
import numpy as np
import sys
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Admin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

plate_image_dimensions_1 = (1200,390)
plate_image_dimensions_2 = (620,480)
box1 = [[0, 0], [620, 0], [620, 480], [0, 480]]

def lpr(image, show_steps = False, show_contour = True, show_plates = True, show_hist = True):
    colorimg = image
    image = cv2.cvtColor(colorimg,cv2.COLOR_BGR2GRAY)
    # pre-process the image
    ppimg = pre_process(image)
    edgeimg = detect_edges(ppimg)

    if show_steps:
        helper_imshow("Original Image", colorimg)
        helper_imshow("Pre-Processed Image", ppimg)
        helper_imshow("Edge Detection Result", edgeimg)
        helper_imwait()

    (ok, out, approx, cnt) = try_get_license_plate(ppimg, edgeimg)

    # show steps if the flag is set
    if show_contour:
        tmp = cv2.cvtColor(ppimg.copy(), cv2.COLOR_GRAY2RGB)
        lst = [out, approx, cnt]
        cv2.drawContours(tmp, lst, 0, (255, 0, 0), 2)
        helper_imshow("License Plate Region", tmp)
        helper_imwait()

    if not ok:
        return "NOT FOUND"

    plate, hist, w, h = separate_resize_plate(ppimg, approx, cnt)
    # make the plate binary
    binplate, thresh = binarize_plate(plate, hist)
    clearplate = remove_plate_details(binplate)

    n_plate = cv2.bitwise_not(plate)
    boxx = find_license_plate_p(n_plate);
    binplate, _ = binarize_plate(n_plate, hist)
    trans_img = transform(boxx, binplate)
    n_clearplate = remove_plate_details(trans_img)
    n_clearplate = cv2.bitwise_not(n_clearplate)
    n_clearplate = pre_process(n_clearplate)

    # dilate and erode image to remove small letters and screws

    # show plate
    if show_plates:
        helper_imshow("plate", plate)
        helper_imshow("binplate", binplate)
        helper_imshow("clearplate", clearplate)
        # helper_imshow("tran", trans_img)
        helper_imshow("n_clear", n_clearplate)


        helper_imwait()

    text = tesseract(clearplate, plate)
    # text = post_process(text)

    return text

def pre_process(image):
    out_min = 2
    out_max = 255
    (in_min, in_max, _, _) = cv2.minMaxLoc(image)
    c_factor = (out_max - out_min) / (in_max - in_min)
    return ((image - in_min) * c_factor + out_min).astype(np.uint8)
    return image

def detect_edges(image):
    image = cv2.bilateralFilter(image, 11, 17, 17)  # Blur to reduce noise
    out = cv2.Canny(image, 50, 270, apertureSize=3, L2gradient=True)
    return out

def try_get_license_plate(image, edgeimg):
    (ok, out, approx, cnt) = find_license_plate(edgeimg)
    if ok:
        return ok, out, approx, cnt

    otsuimg = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    helper_showwait("otsu", otsuimg)
    (ok, out, approx, cnt) = find_license_plate(otsuimg)
    if ok:
        return ok, out, approx, cnt

    adaptimg = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)
    adaptimg = cv2.bitwise_not(adaptimg)
    adaptimg = cv2.dilate(adaptimg, np.ones((3,3), np.uint8), iterations = 1)
    adaptimg = cv2.erode(adaptimg, np.ones((5,5), np.uint8), iterations = 2)
    helper_showwait("adapt", adaptimg)
    (ok, out, approx, cnt) = find_license_plate(adaptimg)
    if ok:
        return ok, out, approx, cnt

    return False, None, None, None

def find_license_plate(image, accepted_ratio_min= 1, accepted_ratio_max = 3,  error = 0.37):
    area_threshold = 500 # arbitrary threshold for plate area in image.
    contours, h = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(h)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.05, True)

        # find contours with 4 edges and the area of which is greater than threshold
        if len(approx) >= 4 and np.abs(cv2.contourArea(contour)) > area_threshold:
            rect = cv2.minAreaRect(contour)
            box = np.intp(cv2.boxPoints(rect))
            (box_w, box_h) = helper_boxwh(box)

            ratio = box_w / box_h

            if accepted_ratio_min - error < ratio and ratio < accepted_ratio_max + error:
                return (True, box, approx, contour)
    return False, None, None, None

def find_license_plate_p(image):
    area_threshold = 500 # arbitrary threshold for plate area in image.
    contours, h = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.05, True)

        # find contours with 4 edges and the area of which is greater than threshold
        if len(approx) >= 4 and np.abs(cv2.contourArea(contour)) > area_threshold:
            rect = cv2.minAreaRect(contour)
            aaa = np.intp(cv2.boxPoints(rect))
            # print(aaa)
    return aaa


def separate_resize_plate(image, apr, cnt,):
    # Create an image containing only the plate
    cleanimg = image.copy()
    mask = np.full_like(cleanimg, 255)
    cv2.drawContours(mask, [cnt], 0, (0, 0, 0), -1)
    cv2.drawContours(mask, [cnt], 0, (255, 255, 255), 2)
    cleanimg = cv2.add(cleanimg, mask)

    # calculate histogram
    ri = cleanimg.ravel()
    rm = mask.ravel()
    hist = np.zeros(256)

    for i in range(len(rm)):
        if rm[i] == 0:
            hist[ri[i]] += 1

    # cumulative histogram
    cumulative = np.zeros_like(hist)
    cumulative[0] = hist[0]
    for i in range(len(cumulative) - 1):
        cumulative[i + 1] = cumulative[i] + hist[i + 1]

    pixels = cumulative[255]

    # equalized image creation
    (w,h) = cleanimg.shape
    clone = cleanimg.copy()
    for i in range(w):
        for j in range(h):
            if mask[i][j] == 0:
                clone[i][j] = np.int8((255 / pixels) * cumulative[cleanimg[i][j]])

    # work on the plate region
    (prx, pry, prw, prh) = cv2.boundingRect(apr)
    plate = cleanimg[pry:pry+prh, prx:prx+prw].copy()

    ratio_p = prw / prh
    # Resize the plate

    if ratio_p > 2.5:
        plate = cv2.resize(plate, plate_image_dimensions_1)
    else:
        plate = cv2.resize(plate, plate_image_dimensions_2)

    return plate, hist, prw, prh

def binarize_plate(plate, hist):
    thresh = calculate_otsu(hist)
    _, result = cv2.threshold(plate, thresh, 255, cv2.THRESH_BINARY)
    return result, thresh
def calculate_otsu(hist):
    nbins = 256
    p = hist / np.sum(hist)
    sigma_b = np.zeros((256,1))
    for t in range(nbins):
        q_L = sum(p[:t])
        q_H = sum(p[t:])
        if q_L == 0 or q_H == 0:
            continue

        miu_L = sum(np.dot(p[:t], np.transpose(np.matrix([i for i in range(t)])))) / q_L
        miu_H = sum(np.dot(p[t:], np.transpose(np.matrix([i for i in range(t, 256)])))) / q_H
        sigma_b[t] = q_L * q_H * (miu_L - miu_H) ** 2

    return np.argmax(sigma_b)


def remove_plate_details(plate):
    result = cv2.dilate(plate, np.ones((5,5), np.uint8), iterations = 2)
    result = cv2.erode(result, np.ones((5,5), np.uint8), iterations = 2)
    return result


# def plate_remove_nonconforming(plate):
#     inverted = cv2.bitwise_not(plate)
#     return cv2.bitwise_not(inverted)

def transform(boxx, img):
    pts1 = np.float32([boxx[1], boxx[2], boxx[3], boxx[0]])
    pts2 = np.float32([[0, 0], [620, 0], [620, 480], [0, 480]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (620, 480))

    return dst




def tesseract(clearplate, plate):
    text = pytesseract.image_to_string(Image.fromarray(clearplate))
    text = ' '.join(text.split())

    if text == None:
        text = pytesseract.image_to_string(Image.fromarray(plate))
        text = ' '.join(text.split())
    return text


#  B == 8, G --6, I -- 1, Z -- 2, O -- 0,
# def post_processing():


def helper_imshow(name, image):
    cv2.imshow(name, image)

def helper_imwait():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def helper_showwait(name, image):
    helper_imshow(name, image)
    helper_imwait()

def helper_boxwh(box):
    x1 = box[0][0]
    y1 = box[0][1]
    x2 = box[1][0]
    y2 = box[1][1]
    x3 = box[2][0]
    y3 = box[2][1]

    w = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    h = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

    if np.abs(y2 - y1) < np.abs(y3 - y2):
        return (w,h)
    else:
        return (h,w)

# Code to initialize an image and print the license plate in it
if len(sys.argv) < 2:
    sys.exit("Usage: lpr.py <filename> [<show_steps>]")

image = cv2.imread(str(sys.argv[1]), cv2.IMREAD_COLOR)

image = cv2.resize(image, (1600, 900))

# Check the show steps argument
show_steps = False
if len(sys.argv) > 2:
    show_steps = (int(sys.argv[2]) == 1)

show_contour = False
if len(sys.argv) > 3:
    show_contour = (int(sys.argv[3]) == 1)

show_plates = False
if len(sys.argv) > 4:
    show_plates = (int(sys.argv[4]) == 1)

show_hist = False
if len(sys.argv) > 5:
    show_hist = (int(sys.argv[5]) == 1)

# print final result (plate's number)
print(lpr(image, show_steps, show_contour, show_plates, show_hist))
