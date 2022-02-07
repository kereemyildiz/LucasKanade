# Author: Ali Kerem Yıldız
# Student Id: 150170013

import cv2
from scipy import signal
import numpy as np
import moviepy.editor as mpy
import os
import re
import glob

# PSEUDECODE OF THE BELOW IMPLEMENTATION

# Read two consecutive frames
# Convert to gray mode and apply GaussianBlur
# Calculate Ix, Iy, It (By using signal.convolve2d)
# Find the appropriate points by using goodFeaturesToTrack
# for all points:
#   Use 3*3 window for the gradient motion_vector(IX, IY, IT)
#   Apply Least Square Solution
#   find motion motion_vector
#   Apply threshold and draw the arrow
# Append that frame to images_list


images_list = []

def LucasKanade(i1,i2, threshold):
    i1_gray = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2_gray = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

    s = i1.shape
    
    i1_smooth = cv2.GaussianBlur(i1_gray, (7,7), 0)
    i2_smooth = cv2.GaussianBlur(i2_gray, (7,7), 0)
    
    Ix = signal.convolve2d(i1_smooth, [[0.25, -0.25], [0.25, -0.25]], 'same') + signal.convolve2d(i2_smooth, [[0.25, -0.25], [0.25, -0.25]], 'same') 
    Iy = signal.convolve2d(i1_smooth, [[0.25, 0.25], [-0.25, -0.25]], 'same') + signal.convolve2d(i2_smooth, [[0.25, 0.25], [-0.25, -0.25]], 'same') 
    It = signal.convolve2d(i1_smooth, [[-0.25, -0.25], [-0.25, -0.25]], 'same') + signal.convolve2d(i2_smooth, [[+0.25, +0.25], [+0.25, +0.25]], 'same')
    
    pts = cv2.goodFeaturesToTrack(i1_smooth, 10000, 0.01, 10)
    
    image = i1_smooth.copy()
    
    pts = np.int0(pts)
    
    for a in pts:
        j, i = a.ravel()
        
        
        IX = ([Ix[i - 1, j - 1], Ix[i, j - 1], Ix[i + 1, j - 1], Ix[i - 1, j], Ix[i, j], Ix[i + 1, j], Ix[i - 1, j + 1],
            Ix[i, j + 1], Ix[i + 1, j + 1]])  
        IY = ([Iy[i - 1, j - 1], Iy[i, j - 1], Iy[i + 1, j - 1], Iy[i - 1, j], Iy[i, j], Iy[i + 1, j], Iy[i - 1, j + 1],
            Iy[i, j + 1], Iy[i + 1, j + 1]]) 
        IT = ([-It[i - 1, j - 1], -It[i, j - 1], -It[i + 1, j - 1], -It[i - 1, j], -It[i, j], -It[i + 1, j], -It[i - 1, j + 1],
            -It[i, j + 1], -It[i + 1, j + 1]]) 



        A = np.array([IX, IY]).T
        A_T = A.T
        A1 = A_T @ A
        A2 = np.linalg.pinv(A1)
                
        B = np.array([IT]).T
        B2 = A_T @ B
        
        
        motion_vector = A2 @ B2
        
        if np.linalg.norm(motion_vector) > threshold:
            motion_vector = motion_vector*10
            motion_vector += np.array([[j],[i]])
            motion_vector = np.int0(motion_vector)
            
            
            image = cv2.arrowedLine(image, (j, i), (motion_vector[0][0], motion_vector[1][0]), color=(255,0,0), thickness=2, tipLength=1)
        
    final_frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    images_list.append(final_frame)

# Prevent randomized file read
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def file_traverse():
    parent_dir = os.getcwd()
    image_dir = parent_dir + '\DJI_0101'
    os.chdir(image_dir)
    print(os.getcwd())
    k = 1
    threshold=1
    
    for file in sorted(glob.glob('*.png'),key=numericalSort):
        
        if file != '00459.png':
            next_file_name = f'{k:05}.png'
            k += 1
            i1 = cv2.imread(file)
            i2 = cv2.imread(next_file_name)
            LucasKanade(i1,i2, threshold)
        
file_traverse()
os.chdir('..')
clip = mpy.ImageSequenceClip(images_list, fps = 20)
file = 'part1.1_video.mp4'
clip.write_videofile(file, codec = 'libx264')

