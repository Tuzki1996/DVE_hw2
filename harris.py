import os
import glob
import cv2
import numpy as np
import scipy.ndimage
from math import *
import argparse


def harris_corner(raw_img, kernel = (9, 9), sigma=3, define_args=False):
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img, kernel, sigma)
    Iy, Ix = np.gradient(blur)

    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyy = Iy ** 2

    Sxx = cv2.GaussianBlur(Ixx, kernel, sigma)
    Sxy = cv2.GaussianBlur(Ixy, kernel, sigma)
    Syy = cv2.GaussianBlur(Iyy, kernel, sigma)

    k = 0.04
    detM = Sxx * Syy - Sxy ** 2
    traceM = Sxx + Syy
    harris_response = detM - k * (traceM ** 2)

    '''
    # option: using summation
    height, width = img.shape
    window_size = 6
    offset = int(window_size/2)
    harris_response = np.zeros(img.shape)
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
            Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])

            det = Sxx * Syy - Sxy ** 2
            trace = Sxx + Syy
            response = det - k * (trace ** 2)

            harris_response[y][x] = response

    cv2.imwrite("./response_2.jpg", harris_response)
    '''

    if define_args:
        if args.save:
            cv2.imwrite(os.path.join(args.output_dir, "gray.jpg"), img)
            cv2.imwrite(os.path.join(args.output_dir, "ix.jpg"), Ix)
            cv2.imwrite(os.path.join(args.output_dir, "iy.jpg"), Iy)
            cv2.imwrite(os.path.join(args.output_dir, "ixx.jpg"), Ixx)
            cv2.imwrite(os.path.join(args.output_dir, "ixy.jpg"), Ixy)
            cv2.imwrite(os.path.join(args.output_dir, "iyy.jpg"), Iyy)
            cv2.imwrite(os.path.join(args.output_dir, "harris_response.jpg"), harris_response)

            # plot corners and edges
            corners, edges = np.copy(raw_img), np.copy(raw_img)
            for r_i, rs in enumerate(harris_response):
                for c_i, r in enumerate(rs):
                    if r > 0:
                        corners[r_i, c_i] = [0, 0, 255]
                    elif r < 0:
                        edges[r_i, c_i] = [0, 255, 0]

            cv2.imwrite(os.path.join(args.output_dir, "corner.jpg"), corners)
            cv2.imwrite(os.path.join(args.output_dir, "edge.jpg"), edges)

    return harris_response


def non_maximum_suppression(img, harris_response, side=10, top_n = 256, define_args=False):
    height, width = harris_response.shape

    max_response = scipy.ndimage.maximum_filter(harris_response, (side, side))
    harris_response = harris_response * (harris_response == max_response)

    corners = np.dstack(np.unravel_index(np.argsort(harris_response.ravel()), (height, width)))[0][::-1]

    # remove close-to-boundary points
    selected_corners = []
    for (h, w) in corners:
        if h > 7 and w > 7 and h < height - 8 and w < width - 8:
            selected_corners.append([w, h])
    selected_corners = np.array(selected_corners)[:top_n]

    if define_args:
        if args.save:
            cv2.imwrite(os.path.join(args.output_dir, "max_response.jpg"), max_response)
            cv2.imwrite(os.path.join(args.output_dir, "harris_response_after.jpg"), harris_response)

            corners_img = np.copy(img)
            for (w, h) in selected_corners:
                cv2.circle(corners_img, (w, h), 4, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(args.output_dir, "corner_after_{}.jpg".format(cnt)), corners_img)

    return selected_corners


def collect_harris_descriptors(img, corners, offset=4):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    descriptors = []
    for (w, h) in corners:
        feature = img[h-offset:h+1+offset, w-offset:w+1+offset].flatten()
        norm = np.linalg.norm(feature)
        feature = feature / norm
        descriptors.append(feature)

    return np.array(descriptors)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default = 'parrington', type = str)
    parser.add_argument("--output_dir", default = 'output', type = str)
    parser.add_argument("--save", action='store_true')
    args = parser.parse_args()

    cnt = 0
    for infile in sorted(glob.glob(os.path.join(args.img_dir, '*.JPG'))):
        print(os.path.basename(infile))
        img = cv2.imread(infile)
        harris_response = harris_corner(img, define_args=True)
        corners = non_maximum_suppression(img, harris_response, define_args=True)
        descriptors = collect_harris_descriptors(img, corners)
        print (corners.shape, descriptors.shape)
        cnt += 1
