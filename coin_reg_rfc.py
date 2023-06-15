import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import imghdr
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Image enhancement
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, 30)
    v = np.clip(v, 0, 255)
    enhanced_hsv = cv2.merge((h, s, v))
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    kernel = np.array([[-1,-1,-1], 
                      [-1,8,-1], 
                      [-1,-1,-1]])
    enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)

    hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 30)
    s = np.clip(s, 0, 255)
    enhanced_hsv = cv2.merge((h, s, v))
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=1.5)

    return enhanced_image

def image_segmentation(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Global Thresholding
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Adaptive Thresholding
    blurred_image = cv2.medianBlur(gray, 5)
    thresh2 = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Canny Edge Detection
    edges = cv2.Canny(gray, 100, 200)

    return thresh1, thresh2, edges

def feature_extraction(image):
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    areas = [cv2.contourArea(cnt) for cnt in contours]
    max_area_index = np.argmax(areas)
    largest_contour = contours[max_area_index]
    
    # Calculate perimeter
    perimeter = cv2.arcLength(largest_contour, True)

    return perimeter, areas[max_area_index]

def preprocess_images(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    for entry in os.scandir(input_directory):
        if entry.is_file():
            # Check if the file is an image
            image_type = imghdr.what(entry.path)
            if image_type in ['jpeg', 'png']:
                # Load and preprocess the image
                image = cv2.imread(entry.path)
                preprocessed_image = preprocess_image(image)

                # Save the preprocessed image to the output directory
                filename = os.path.splitext(entry.name)[0]
                output_path = os.path.join(output_directory, f'{filename}_preprocessed.jpg')
                cv2.imwrite(output_path, preprocessed_image)

                print('Preprocessed image saved:', output_path)

def process_single_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    preprocessed_image = preprocess_image(image)

    # Perform image segmentation
    thresh1, thresh2, edges = image_segmentation(preprocessed_image)

    # Perform feature extraction
    perimeter, area = feature_extraction(thresh1)

    return perimeter, area

def process_images_with_labels(input_directory):
    features = []
    labels = []

    for entry in os.scandir(input_directory):
        if entry.is_file():
            # Check if the file is an image
            image_type = imghdr.what(entry.path)
            if image_type in ['jpeg', 'png']:
                # Extract the class label from the image file name
                image_name = os.path.splitext(entry.name)[0]
                
                if image_name == 'C' or image_name == 'Coins':
                    class_label = 'mixture'
                elif '20c' in image_name:
                    class_label = '20c'
                elif '10c' in image_name:
                    class_label = '10c'
                elif 'R1' in image_name:
                    class_label = 'R1'
                elif 'R2' in image_name:
                    class_label = 'R2'
                elif 'R5' in image_name or '55' in image_name :
                    class_label = 'R5'
                else:
                    class_label = 'unknown'

                # Preprocess the image
                image = cv2.imread(entry.path)
                preprocessed_image = preprocess_image(image)

                # Perform image segmentation
                thresh1, _, _ = image_segmentation(preprocessed_image)

                # Perform feature extraction
                perimeter, area = feature_extraction(thresh1)

                # Add the features and label to the lists
                features.append([perimeter, area])
                labels.append(class_label)

    return np.array(features), np.array(labels)

def train_and_evaluate(features, labels):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=50)

    # Train the random forest classifier
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred_rf = rf_classifier.predict(X_test)

    # Calculate the accuracy of the random forest classifier
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print('Random Forest Accuracy:', rf_accuracy)

# Set input and output directories
input_directory = r"C:\Users\216018136\Documents\hounors\COMP702\Coins\images"
output_directory = r"C:\Users\216018136\Documents\hounors\COMP702\Coins\images\preprocessed_images"

# Preprocess images in the input directory
preprocess_images(input_directory, output_directory)

# Process preprocessed images with labels
features, labels = process_images_with_labels(output_directory)

# Train and evaluate the classifier
train_and_evaluate(features, labels)
