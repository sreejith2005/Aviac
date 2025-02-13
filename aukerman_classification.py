import numpy as np
import cv2
from sklearn.cluster import KMeans
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
import os

class OrthoImageClassifier:
    def __init__(self, input_shape=(256, 256)):
        self.input_shape = input_shape
        self.kmeans = None
        self.scaler = StandardScaler()
        
    def _extract_features(self, image):
        """Extract enhanced color and texture features from image"""
        # Convert to float32 for better precision
        image = image.astype(np.float32) / 255.0
        
        # Color space conversions
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Calculate vegetation index (NDVI-like)
        eps = 1e-8
        vegetation_index = (image[:,:,1] - image[:,:,0]) / (image[:,:,1] + image[:,:,0] + eps)
        
        # Calculate water index
        water_index = (image[:,:,1] - image[:,:,2]) / (image[:,:,1] + image[:,:,2] + eps)
        
        # Calculate texture features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute gradients
        dx = ndimage.sobel(gray, 1)
        dy = ndimage.sobel(gray, 0)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        
        # Local statistics
        local_std = ndimage.generic_filter(gray, np.std, size=5)
        local_entropy = ndimage.generic_filter(gray, self._entropy, size=5)
        
        # Stack all features
        features = np.dstack([
            image,                    # RGB
            hsv,                      # HSV
            lab,                      # LAB
            vegetation_index[..., np.newaxis],
            water_index[..., np.newaxis],
            gradient_magnitude[..., np.newaxis],
            local_std[..., np.newaxis],
            local_entropy[..., np.newaxis]
        ])
        
        return features
    
    def _entropy(self, values):
        """Calculate local entropy"""
        hist = np.histogram(values, bins=20)[0]
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def _create_superpixels(self, image, n_segments=100):
        """Create superpixels for more coherent regions"""
        try:
            from skimage.segmentation import slic
            segments = slic(image, n_segments=n_segments, compactness=10, sigma=1)
            return segments
        except ImportError:
            print("Warning: scikit-image not available, falling back to regular grid")
            return self._create_grid(image)
    
    def _create_grid(self, image, size=16):
        """Create a regular grid if superpixels are not available"""
        h, w = image.shape[:2]
        segments = np.zeros((h, w), dtype=np.int32)
        for i in range(0, h, size):
            for j in range(0, w, size):
                segments[i:i+size, j:j+size] = (i//size) * (w//size) + (j//size)
        return segments
    
    def _extract_region_features(self, image, segments):
        """Extract features for each superpixel/region"""
        features = self._extract_features(image)
        n_segments = segments.max() + 1
        region_features = []
        
        for i in range(n_segments):
            mask = segments == i
            if mask.sum() > 0:
                # Extract region statistics
                region_stats = []
                for j in range(features.shape[2]):
                    feature_channel = features[:,:,j][mask]
                    region_stats.extend([
                        np.mean(feature_channel),
                        np.std(feature_channel),
                        np.percentile(feature_channel, 25),
                        np.percentile(feature_channel, 75)
                    ])
                region_features.append(region_stats)
        
        return np.array(region_features)
    
    def learn_from_dataset(self, dataset_path, n_clusters=3):
        """Learn from the dataset using region-based approach"""
        dataset_path = Path(dataset_path)
        image_files = list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.png'))
        print(f"Found {len(image_files)} images")
        
        all_features = []
        for i, img_path in enumerate(image_files):
            try:
                print(f"Processing image {i+1}/{len(image_files)}")
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.input_shape)
                
                # Create regions and extract features
                segments = self._create_superpixels(image)
                features = self._extract_region_features(image, segments)
                all_features.extend(features)
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        
        all_features = np.array(all_features)
        print(f"Extracted {len(all_features)} region features")
        
        # Scale features and perform clustering
        all_features_scaled = self.scaler.fit_transform(all_features)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans.fit(all_features_scaled)
        
        return self
    
    def classify_image(self, image_path):
        """Classify regions in a new image"""
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_shape)
        
        # Create regions and extract features
        segments = self._create_superpixels(image)
        features = self._extract_region_features(image, segments)
        
        # Predict clusters
        features_scaled = self.scaler.transform(features)
        predictions = self.kmeans.predict(features_scaled)
        
        # Create segmentation map
        segmentation = np.zeros_like(segments)
        for i, pred in enumerate(predictions):
            segmentation[segments == i] = pred
        
        # Post-process to remove noise
        segmentation = ndimage.median_filter(segmentation, size=3)
        
        # Identify classes
        class_mapping = self._identify_classes(image, segmentation)
        
        # Create visualization
        visualization = self._create_visualization(segmentation, class_mapping)
        
        return segmentation, class_mapping, visualization
    
    def _identify_classes(self, image, segmentation):
        """Identify classes based on domain knowledge"""
        mean_colors = []
        mean_indices = []
        
        for i in range(len(self.kmeans.cluster_centers_)):
            mask = segmentation == i
            if mask.sum() > 0:
                region = image[mask]
                mean_colors.append(np.mean(region, axis=0))
                
                # Calculate vegetation index for the region
                green = region[:,1]
                red = region[:,0]
                veg_index = np.mean((green - red)/(green + red + 1e-8))
                mean_indices.append(veg_index)
        
        # Assign classes based on color and vegetation index
        class_mapping = {}
        indices = list(range(len(mean_colors)))
        
        # Sort by vegetation index
        indices.sort(key=lambda x: mean_indices[x], reverse=True)
        
        # Assign classes
        if len(indices) >= 3:
            class_mapping[indices[0]] = 'vegetation'  # Highest vegetation index
            class_mapping[indices[-1]] = 'water'      # Lowest vegetation index
            class_mapping[indices[1]] = 'land'        # Middle value
        
        return class_mapping
    
    def _create_visualization(self, segmentation, class_mapping):
        """Create visualization with smoother boundaries"""
        color_map = {
            'vegetation': [0, 255, 0],  # Green
            'water': [0, 0, 255],      # Blue
            'land': [139, 69, 19]      # Brown
        }
        
        visualization = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
        for cluster_id, class_name in class_mapping.items():
            mask = segmentation == cluster_id
            visualization[mask] = color_map[class_name]
        
        # Smooth boundaries
        visualization = cv2.GaussianBlur(visualization, (3,3), 0)
        
        return visualization

def train_and_evaluate(dataset_path, test_image_path):
    """Helper function to train and evaluate the model"""
    classifier = OrthoImageClassifier()
    classifier.learn_from_dataset(dataset_path)
    segmentation, class_mapping, visualization = classifier.classify_image(test_image_path)
    
    # Calculate percentages
    total_pixels = segmentation.size
    percentages = {}
    for cluster_id, class_name in class_mapping.items():
        pixels = np.sum(segmentation == cluster_id)
        percentage = (pixels / total_pixels) * 100
        percentages[class_name] = round(percentage, 2)
    
    return classifier, segmentation, visualization, percentages

# Specify paths
dataset_path = r"C:\Users\LENOVO\Downloads\aukerman_classification\images"
test_image_path = r"C:\Users\LENOVO\Downloads\aukerman_classification\test_images\DSC00275.JPG"

try:
    # Train and evaluate
    classifier, segmentation, visualization, percentages = train_and_evaluate(dataset_path, test_image_path)
    
    # Display results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.imread(test_image_path)[...,::-1])
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("Classification Result")
    plt.show()
    
    print("Land cover percentages:", percentages)
    
except Exception as e:
    print(f"Error: {str(e)}")