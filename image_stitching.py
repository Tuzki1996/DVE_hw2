import os
import glob
from math import *
import argparse
import numpy as np
from numpy.linalg import norm
from numpy.random import randint
import cv2

from harris import *


def detect_harris_feature(img):
    harris_response = harris_corner(img)
    corners = non_maximum_suppression(img, harris_response)
    descriptors = collect_harris_descriptors(img, corners)

    return corners, descriptors


def match_feature(des1, des2, kp1, kp2):
    distance = np.zeros([des1.shape[0], des2.shape[0]])

    for i, d1 in enumerate(des1):
        for j, d2 in enumerate(des2):
            distance[i][j] = norm(d1-d2)

    sorted_dist_args = np.argsort(distance)
    top1_arg = sorted_dist_args[:,0]
    top2_arg = sorted_dist_args[:,1]

    top1 = distance[range(distance.shape[0]), top1_arg]
    top2 = distance[range(distance.shape[0]), top2_arg]
    ratio = top1/top2
    valid_points = np.squeeze(np.argwhere(ratio < 0.8))
    point_pairs = np.stack((valid_points, top1_arg[valid_points]), 0).T

    matches = []
    for i,pp in enumerate(point_pairs):
        matches.append([[kp1[pp[0]][0],kp1[pp[0]][1]] , [kp2[pp[1]][0], kp2[pp[1]][1]]])

    return np.array(matches)

def match_image(img1, img2):
    kp1, des1 = detect_harris_feature(img1)
    kp2, des2 = detect_harris_feature(img2)
    matches = match_feature(des1, des2, kp1, kp2)

    return matches

def ransac(point_pairs, sucess_prob = 0.99, inlier_prob = 0.5, n =1):
    k = ceil(log(1-sucess_prob)/log(1-inlier_prob**n))
    vote_list = np.zeros([k])
    model_list = []
    inlier_list = []
    for trial_idx in range(k):
        selected_idx = randint(0, point_pairs.shape[0], n)
        selected_samples = point_pairs[selected_idx]

        # fit model
        m1 = selected_samples[0][0][0]- selected_samples[0][1][0]
        m2 = selected_samples[0][0][1]- selected_samples[0][1][1]

        # store model params
        model_list.append((m1,m2))

        # vote
        inlier = []
        for pp in point_pairs:
            dist = sqrt((m1-pp[0][0]+pp[1][0])**2 + (m2-pp[0][1]+pp[1][1])**2)
            if dist < 10:
                vote_list[trial_idx] += 1
                inlier.append(pp)

        inlier_list.append(inlier)

    best_trial_idx = np.argmax(vote_list)
    return np.array(inlier_list[best_trial_idx])

def solve_translation_matrix_params(point_pairs):
    # sum(i->n) [(m1 + yi - y'i)^2 + m2 + xi - x'i)^2]

    n = point_pairs.shape[0]
    sum_ydiff = 0
    sum_xdiff = 0
    for pp in point_pairs:
        p1 = pp[0]
        p2 = pp[1]

        x_diff = p1[0]-p2[0]
        y_diff = p1[1]-p2[1]

        sum_xdiff += x_diff
        sum_ydiff += y_diff


    m1 = sum_xdiff/n
    m2 = sum_ydiff/n

    return m1,m2

def align_and_blend(imgs, trans_matrix_list):
    x_bounds = [int(imgs[0].shape[1])]
    y_bounds= [int(imgs[0].shape[0])]
    x_starts = [0]
    y_starts = [0]

    img_shape_x = int(imgs[0].shape[1])
    img_shape_y = int(imgs[0].shape[0])

    for m in trans_matrix_list:
        x_bounds.append(int(x_bounds[-1]+m[0]))
        y_bounds.append(int(y_bounds[-1]+m[1]))
        x_starts.append(int(x_bounds[-1]-img_shape_x))
        y_starts.append(int(y_bounds[-1]-img_shape_y))

    panaroma_img = np.zeros([max(y_bounds)-min(y_starts), max(x_bounds), 3], dtype =  np.uint16)

    for idx in range(len(imgs)):

        img_mask = np.ones([img_shape_x])
        pan_mask = np.ones([max(x_bounds)])

        img_o_s = 0
        img_o_e = 0


        pan_o_s = x_starts[idx]
        pan_o_e = 0
        if idx != 0:
            pan_o_e = x_bounds[idx-1]
            img_o_e = x_bounds[idx-1]-x_starts[idx]

        for i in range(img_mask.shape[0]):
            if i >= img_o_s and i < img_o_e:
                img_mask[i] = img_mask[i]*(i/(img_o_e-img_o_s))

        for i in range(pan_mask.shape[0]):
            if i >= pan_o_s and i < pan_o_e:
                pan_mask[i] = pan_mask[i]*(((pan_o_e-i))/(pan_o_e-pan_o_s))


        img_mask = img_mask.reshape(1,img_mask.shape[0],1)
        img_mask = np.tile(img_mask,(imgs[0].shape[0],1,imgs[0].shape[2]))
        masked_img = (imgs[idx]*img_mask).astype(np.uint16)

        pan_mask = pan_mask.reshape(1,pan_mask.shape[0],1)
        pan_mask = np.tile(pan_mask,(panaroma_img.shape[0],1,panaroma_img.shape[2]))
        panaroma_img = (panaroma_img*pan_mask).astype(np.uint16)


        panaroma_img[y_starts[idx]-min(y_starts):y_starts[idx]+img_shape_y-min(y_starts),
                     x_starts[idx]:x_starts[idx]+img_shape_x] += masked_img


    panaroma_img = np.clip(panaroma_img,0,255).astype(np.uint16)
    cropped_img = panaroma_img[max(y_starts)-min(y_starts): min(y_bounds), min(x_starts):max(x_bounds), :]

    return cropped_img




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default = 'parrington_projected', type = str)
    parser.add_argument("--ext", default = 'JPG', type = str)
    parser.add_argument("--output_dir", default = 'output', type = str)
    parser.add_argument("--file_name", default = 'panaroma', type = str)
    parser.add_argument("--reverse", default = False, type = bool)
    parser.add_argument("--save", action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok = True)

    proj_imgs = []
    for infile in sorted(glob.glob(os.path.join(args.img_dir, '*.{}'.format(args.ext)))):
         proj_imgs.append(cv2.imread(infile))
    if args.reverse:
        proj_imgs.reverse()

    trans_matrix_list = []

    for i in range(len(proj_imgs)-1):
        img1 = proj_imgs[i]
        img2 = proj_imgs[i+1]
        point_pairs = match_image(img1, img2)
        filterd_point_pairs = ransac(point_pairs)
        m1, m2 = solve_translation_matrix_params(filterd_point_pairs)
        trans_matrix_list.append([m1, m2])

    panaroma_img = align_and_blend(proj_imgs, trans_matrix_list)
    cv2.imwrite(os.path.join(args.output_dir,args.file_name+".jpg"), panaroma_img)
