import random
import math
import cv2
from scipy.spatial import distance

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
random.seed(0)
 
def distancia(x1, x2):
    #return np.sqrt(np.sum((x1 - x2)**2))
    return distance.euclidean(x1, x2)

imageOriginal = cv2.imread("Jit1.jpg")
rowsOriginal = imageOriginal.shape[0]
colsOriginal = imageOriginal.shape[1]
print(imageOriginal.shape) # Print image shape

image = cv2.imread("Jit_resized.jpg")
rowsResized = image.shape[0]
colsResized = image.shape[1]
print(image.shape) # Print image shape
cv2.imshow("original", image)

factorResized = rowsOriginal / rowsResized
print("\nFactor de cambio de tamano entre la imagen original y la imagen resized: ", factorResized)

 
# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1,3)).copy()

# Convert to float type
pixel_vals = np.float32(pixel_vals)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

k = 7
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
 
# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
 
# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
 
cv2.imshow("segmented_image", segmented_image)

segmented_image2 = image.reshape((-1,3)).copy()

listaCluster = []
for i in range(k):
    listaCluster.append(i)
#print(listaCluster)

kRojo = 0

for j in listaCluster:
    if j != kRojo:
        segmented_image2[labels.flatten() == j] = np.uint8(np.array([0, 0, 0]))
segmented_image2 = segmented_image2.reshape((image.shape))


cv2.imshow("cluster rojo", segmented_image2)

t_lower = 50  # Lower Threshold
t_upper = 150  # Upper threshold
  
# Applying the Canny Edge filter
edge = cv2.Canny(segmented_image2, t_lower, t_upper)
  
cv2.imshow('edge', edge)


# Cropping an image
cropped_image1 = edge[40:130, 165:255]
cropped_image2 = edge[180:270, 100:190]
 
# Display cropped image
cv2.imshow("cropped1", cropped_image1)
cv2.imshow("cropped2", cropped_image2)

rows1,cols1 = cropped_image1.shape
rows2,cols2 = cropped_image2.shape

# print(cropped_image1.shape)
# print(cropped_image2.shape)

print(rows1,cols1)


coordenadasBordes1 = []
coordenadasBordes2 = []
for x in range(rows1):
    for y in range(cols1):
        if cropped_image1[x,y] == 255:  
            coordenadasBordes1.append((x,y))

for x in range(rows2):
    for y in range(cols2):
        if cropped_image2[x,y] == 255:  
            coordenadasBordes2.append((x,y))
        
# print(coordenadasBordes1)
# print("\n")
# print(coordenadasBordes2) 

min_tuple1 = min(coordenadasBordes1, key=lambda tup: tup[0])
max_tuple1 = max(coordenadasBordes1, key=lambda tup: tup[0])

min_tuple2 = min(coordenadasBordes2, key=lambda tup: tup[0])
max_tuple2 = max(coordenadasBordes2, key=lambda tup: tup[0])

# print(min_tuple1, max_tuple1)  
# print(min_tuple2, max_tuple2)  

# Window name in which image is displayed
window_name = 'Imagen final'


start_point1 = (160+min_tuple1[0]-3, 43+min_tuple1[1])
end_point1 = (160+max_tuple1[0]+13, 40+min_tuple1[1])

start_point2 = (100+min_tuple2[0]+6, 180+min_tuple2[1]+12)
end_point2 = (100+max_tuple2[0]-8, 180+max_tuple2[1]-8)

print("\nPunto minimo imagen 1: (", int(start_point1[0] * factorResized),"," , int(start_point1[1] * factorResized),")")
print("Punto maximo imagen 1: (", int(end_point1[0] * factorResized),"," , int(end_point1[1] * factorResized),")")

print("\nPunto minimo imagen 2: (", int(start_point2[0] * factorResized),", " , int(start_point2[1] * factorResized),")")
print("Punto maximo imagen 2: (", int(end_point2[0] * factorResized),", " , int(end_point2[1] * factorResized),")")

print("\nLongitud jitomate 1: ", distancia(start_point1, end_point1) * factorResized)
print("Longitud jitomate 2: ", distancia(start_point2, end_point2) * factorResized)


color = (255, 0, 0)
thickness = 1

image = cv2.line(image, start_point1, end_point1, color, thickness)
image = cv2.line(image, start_point2, end_point2, color, thickness)
 
# Displaying the image
cv2.imshow(window_name, image)

# Save the cropped image
#cv2.imwrite("Cropped Image.jpg", cropped_image)
 
cv2.waitKey(0)
cv2.destroyAllWindows()