import cv2
import os
import numpy as np


recognizer = cv2.face.EigenFaceRecognizer.create() # Creating eigenface classifier
detector= cv2.CascadeClassifier("Frameworks_ML_IoT\Face_Recognition\classificadores\haarcascade-frontalface-default.xml")


def get_img_with_id(data_folder, target_size=(100, 100), augment=False):
    images = []
    labels = []
    label = 0

    # Loop through each subfolder (one per person)
    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder)

        if not os.path.isdir(subfolder_path):
            continue
        
        if str(subfolder).lower() == "others":
            # Handle "others" class separately
            for image_name in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                img = cv2.resize(img, target_size)

                # Data augmentation (optional)
                if augment:
                    # Flip the images
                    flipped_img = cv2.flip(src=img, flipCode=0)
                    images.append(flipped_img)
                    labels.append(label)
                    
                    # Routate the images
                    routated_img = cv2.rotate(src=img, rotateCode=0)
                    images.append(routated_img)
                    labels.append(label)
                    
                    # Apply noise (Gaussian noise)
                    noisy_img = add_gaussian_noise(img)
                    images.append(noisy_img)
                    labels.append(label)

                    # Change brightness
                    brightened_img = change_brightness(img, alpha=1.5, beta=20)
                    images.append(brightened_img)
                    labels.append(label)

                images.append(img)
                labels.append(label)
        else:
            # Handle recognized individuals
            for image_name in os.listdir(subfolder_path):
                # Load the image
                image_path = os.path.join(subfolder_path, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                # Resize the image to the target size
                img = cv2.resize(img, target_size)

                # Data augmentation (optional)
                if augment:
                    # Flip the images
                    flipped_img = cv2.flip(src=img, flipCode=0)
                    images.append(flipped_img)
                    labels.append(label)
                    
                    # Routate the images
                    routated_img = cv2.rotate(src=img, rotateCode=0)
                    images.append(routated_img)
                    labels.append(label)
                    
                    # Apply noise (Gaussian noise)
                    noisy_img = add_gaussian_noise(img)
                    images.append(noisy_img)
                    labels.append(label)

                    # Change brightness
                    brightened_img = change_brightness(img, alpha=1.5, beta=20)
                    images.append(brightened_img)
                    labels.append(label)

                # Append the image and label to the lists
                images.append(img)
                labels.append(label)

        label += 1
    print(f'np.array(labels): {np.array(labels)}\n\n images: {images}')

    return np.array(labels), images


# Function to add Gaussian noise to an image
def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


# Function to change image brightness and contrast
def change_brightness(image, alpha=1.0, beta=0):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


# Define data folder and whether to use data augmentation
data_folder = "Frameworks_ML_IoT//Face_Recognition//fotos"
use_data_augmentation = True

ids, faces = get_img_with_id(data_folder, augment=use_data_augmentation)

print("Treinando...")
recognizer.train(faces, ids)
recognizer.save('my_Eigen_classifier.yml')

print("Treinamento realizado!")
