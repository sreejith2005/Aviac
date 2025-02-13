import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image_path = r"C:\Users\LENOVO\Downloads\Computer-Vision-for-Plant-Counting-\images\Count1.tif"  # Change this to your image path
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5,5), 0)

# Adaptive Thresholding for better segmentation
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Morphological operations to separate plants
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Distance Transform to identify plant centers
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

# Find sure background by dilation
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labeling
_, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that the sure background is not 0
markers = markers + 1
markers[unknown == 255] = 0

# Apply Watershed algorithm
image_copy = image.copy()
cv2.watershed(image_copy, markers)
image_copy[markers == -1] = [0, 0, 255]  # Mark boundaries in red

# Find contours again after watershed
contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Display results
plt.figure(figsize=(12,6))
plt.subplot(1,3,1), plt.imshow(gray, cmap="gray"), plt.title("Grayscale")
plt.subplot(1,3,2), plt.imshow(thresh, cmap="gray"), plt.title("Thresholded")
plt.subplot(1,3,3), plt.imshow(contour_image), plt.title(f"Detected Plants: {len(contours)}")
plt.show()
