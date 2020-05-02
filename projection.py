import os
import glob
import cv2
import numpy as np
from math import *
import matplotlib.pyplot as plt
import argparse

def cylindrical_projection(img, f, s):
    proj_x = np.zeros([img.shape[0],img.shape[1]],dtype = int)
    proj_y = np.zeros([img.shape[0],img.shape[1]],dtype = int)
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            
            c_y = y - img.shape[0]/2
            c_x = x - img.shape[1]/2
            
            h = c_y/sqrt(c_x**2 + f**2)
            theta = atan(c_x/f)
            proj_x[y][x] = s*theta
            proj_y[y][x] = s*h
    
    proj_img = np.zeros([ceil(2*proj_y.max())+1, ceil(2*proj_x.max())+1, 3],dtype = np.uint8)
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):            
            proj_img[proj_y[y][x]+proj_y.max(), proj_x[y][x]+proj_x.max(),:] = img[y,x,:]

    return proj_img

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()  
    parser.add_argument("--img_dir", default = 'parrington', type = str)
    parser.add_argument("--f", default = 704.916, type = float)
    args = parser.parse_args()
    
    output_dir = args.img_dir + "_projected"
    os.makedirs(output_dir, exist_ok = True)
    
    for infile in sorted(glob.glob(os.path.join(args.img_dir, 'prtn*.jpg'))):
        print(os.path.basename(infile))
        img = cv2.imread(infile)
        proj_img = cylindrical_projection(img, args.f, args.f)
        cv2.imwrite(os.path.join(output_dir,os.path.basename(infile)), proj_img) 
