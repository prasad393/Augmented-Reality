#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2
import tensorflow as tf
import os
import numpy as np
from cv2 import aruco
import math
import matplotlib.pyplot as plt

image = cv2.imread('C:/Users/VISHAL.000/Desktop/opencv/pictures/20221115_113424.jpg')
image2 = cv2.imread('C:/Users/VISHAL.000/Desktop/opencv/pictures/poster.jpg')

def image_shower(image):
    cv2.namedWindow('window',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('window',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#image_shower(image)

height, width,chhanle= image2.shape
c1=[0,0]
c2=[width,0]
c3=[width,height]
c4=[0,height]

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
parameters = aruco.DetectorParameters_create()

rcorners, ids,Error = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

rc1=rcorners[0][0][0]
rc2=rcorners[0][0][1]
rc3=rcorners[0][0][2]
rc4=rcorners[0][0][3]

#original_corners = [rc1,rc2,rc3,rc4]
original_corners = np.array( [rc1,rc2,rc3,rc4], dtype=np.float32)

s = 4

center = np.mean(original_corners, axis=0)
translated_corners = original_corners - center
scaled_corners = translated_corners * s
final_corners = scaled_corners + center

# Print the original and scaled corner points
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(original_corners[:, 0], original_corners[:, 1], '-o')
ax[0].set_title('Original Shape')
ax[0].invert_yaxis() 
ax[1].plot(final_corners[:, 0], final_corners[:, 1], '-o')
ax[1].set_title('Scaled Shape')
ax[1].invert_yaxis() 
plt.show()


# In[14]:


fc1=[final_corners[0,0],final_corners[0,1]]
fc2=[final_corners[1,0],final_corners[1,1]]
fc3=[final_corners[2,0],final_corners[2,1]]
fc4=[final_corners[3,0],final_corners[3,1]]


# In[ ]:


input_corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
output_corners = np.array([[u1, v1], [u2, v2], [u3, v3], [u4, v4]], dtype=np.float32)
src_homogeneous = tf.pad(src_corners, paddings=[[0, 0], [0, 1]])
dst_homogeneous = tf.pad(dst_corners, paddings=[[0, 0], [0, 1]])
perspective_matrix = tf.linalg.inv(tf.linalg.matmul(dst_homogeneous, tf.linalg.pinv(src_homogeneous)))
with tf.Session() as sess:
    print("Perspective Matrix:")
    print(sess.run(perspective_matrix))


# In[16]:


transformed_image2= cv2.warpPerspective(image2, M,(image.shape[1], image.shape[0]))
image_shower(transformed_image2)
cv2.imwrite('spk.png',transformed_image2)


# In[17]:


pts = np.array([fc1,fc2,fc3,fc4], np.int32)

# Create mask with zeros
mask = np.zeros_like(image)
cv2.fillPoly(mask, [pts], (255, 255, 255))

# Invert mask to keep everything outside the polygon area
mask = cv2.bitwise_not(mask)

# Apply mask to image to remove the polygon area
image = cv2.bitwise_and(image, mask)

final_image = cv2.bitwise_or(image, transformed_image2)
image_shower(final_image)
cv2.imwrite('C:/Users/VISHAL.000/Desktop/opencv/pictures/spk.png',final_image)

