import sys
o_path = "./"
sys.path.append(o_path)

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import x_final
import argparse
import csv


parser = argparse.ArgumentParser()

parser.add_argument('--image_name', type=str, help='image file name, xxx.png', default='111_HC.png')
parser.add_argument('--file_path', type=str, help='directory of where image file is saved', default='datasets\\training_set\\')
parser.add_argument('--csv_file_path', type=str, help='csv file path', default='HC_18_Challenge\\training_set_pixel_size_and_HC.csv')
parser.add_argument('--compute_method', type=str, choices=['ellipse', 'direct'], help='method to compute circumference', default='ellipse')
parser.add_argument('--visualization', type=bool, help='display circumference image or not', default=True)

args = parser.parse_args()


def generate_ellipse(classed_img, img2show):
    # create an image for visualization(just for visualization!)
    # img2show = x_final.label_to_color_image(classed_img).astype(np.uint8)
    cp_image = img2show.copy()

    # extract object region
    target_y = np.where(classed_img != 0)[0]
    target_x = np.where(classed_img != 0)[1]
    target = []
    for i in range(len(target_x)):
        target.append([target_x[i], target_y[i]])

    # extract minimum bounding box
    rect = cv.minAreaRect(np.array(target))
    center = tuple(np.array(rect[0]).astype(np.int))
    axis = tuple((np.array(rect[1]).astype(np.int)/2).astype(np.int))
    angle = np.array(rect[2]).astype(int)

    # draw bounding box
    # box = cv.boxPoints(rect)
    # box = np.int0(box)
    # cv.drawContours(img2show, [box], 0, (0, 0, 255), 2)

    # draw ellipse
    final_img = cv.ellipse(cp_image, center, axis, angle, 0, 360, (255, 255, 0))

    # compute circumference
    cir = 2*np.pi*np.min(axis) + 4*np.abs(axis[0] - axis[1])

    return final_img, cir


def cir_direct(classed_img, img2show):
    # create an image for visualization(just for visualization!)
    # img2show = x_final.label_to_color_image(classed_img).astype(np.uint8)
    cp_img = img2show.copy()

    # extract the number of bounding pixels
    contour, hier = cv.findContours(classed_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    convert_contour = np.array(contour)
    cir = convert_contour.shape[1]

    # draw contour
    final_img = cv.drawContours(cp_img, contour, 0, (255, 255, 0))
    return final_img, cir


def display_result(img, label_image, img_with_contour):
    plt.figure()

    plt.subplot(131)
    plt.imshow(img)
    plt.imshow(label_image, alpha=0.7)
    plt.title("original image with mask")

    plt.subplot(132)
    plt.imshow(label_image)
    plt.title("mask")

    plt.subplot(133)
    plt.imshow(img_with_contour)
    plt.title("contour")

    plt.show()


if __name__ == "__main__":
    TEST_IMAGE_NAME = args.image_name
    TEST_IMAGE_PATH = args.file_path + TEST_IMAGE_NAME
    original_img = cv.imread(TEST_IMAGE_PATH, cv.IMREAD_GRAYSCALE)

    csv_path = args.csv_file_path
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        res = np.array(list(reader))
        pos = np.array(np.where(res == TEST_IMAGE_NAME))[0][0]
        pix_length = res[pos][1].astype(np.float)
        ref_circumference = res[pos][2].astype(np.float)


    classed_img = x_final.run_image(TEST_IMAGE_PATH)
    classed_img = classed_img.astype(np.uint8)
    img2show = x_final.label_to_color_image(classed_img).astype(np.uint8)

    cal_method = args.compute_method
    if cal_method == 'ellipse':
        img2show_elli, cir_ellipse = generate_ellipse(classed_img, img2show)
        print("bounding ellipse contour pixels: %d, approximate circumference: %2f, reference circumference: %2f" % (cir_ellipse.astype(np.int), cir_ellipse.astype(np.int) * pix_length, ref_circumference))
        if args.visualization:
            display_result(original_img, img2show, img2show_elli)
    else:
        img2show_obj, cir_object = cir_direct(classed_img, img2show)
        print("object contour pixels: %d, , approximate circumference: %2f, reference circumference: %2f" % (cir_object, cir_object * pix_length, ref_circumference))
        if args.visualization:
            display_result(original_img, img2show, img2show_obj)

    img_name = "./seg_" + TEST_IMAGE_NAME
    cv.imwrite(img_name, classed_img)
