import os
import cv2
import numpy as np
from tqdm import tqdm
import random
IMAGE_DIR = r"F:\all\24_02\Test_data"


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def add_minimal_noise(image):


    level = random.choice([0, 1, 2])
    if level == 0:
        noise = np.random.randint(-10, 11, image.shape, dtype=np.int16)
    elif level == 1:
        noise = np.random.randint(-5, 6, image.shape, dtype=np.int16)
    else: 
        noise = np.random.randint(-2, 3, image.shape, dtype=np.int16)
    noisy_image = image.astype(np.int16) + noise

    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, strength=2):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened



def apply_clahe_color(image):
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    # Convert back to BGR color space
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def test_enhancement(image):

    clahe_enhanced = apply_clahe_color(image)

    gamma_corrected = adjust_gamma(clahe_enhanced, gamma=0.8)
    

    final_transformed = unsharp_mask(gamma_corrected, kernel_size=(5, 5), sigma=1.0, strength=2)
    
    return final_transformed

def augment_image(image, apply_test_enhancement=False):
    if apply_test_enhancement:
        return test_enhancement(image)
    
    augmented = image.copy()
    augmentation_applied = False

    if np.random.rand() < 0.3:
        gamma = np.random.uniform(0.7, 1.25)
        augmented = adjust_gamma(augmented, gamma)
        augmentation_applied = True

    if np.random.rand() < 0.3:
        brightness = np.random.uniform(0.8, 1.05)
        augmented = cv2.convertScaleAbs(augmented, alpha=brightness, beta=0)
        augmentation_applied = True

    if np.random.rand() < 0.3:
        contrast = np.random.uniform(0.8, 1.2)
        augmented = cv2.convertScaleAbs(augmented, alpha=contrast, beta=0)
        augmentation_applied = True

    if np.random.rand() < 0.3:
        kernel_size = np.random.choice([3, 5])
        augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), 0)
        augmentation_applied = True

    if np.random.rand() < 0.3:
        augmented = add_minimal_noise(augmented)
        augmentation_applied = True

    if np.random.rand() < 0.4:
        augmented = cv2.flip(augmented, 1)
        augmentation_applied = True

    if np.random.rand() < 0.4:
        augmented = cv2.flip(augmented, 0)
        augmentation_applied = True

    if np.random.rand() < 0.15:
        angle = np.random.choice([90, 180, 270])
        rows, cols = augmented.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        augmented = cv2.warpAffine(augmented, M, (cols, rows))
        augmentation_applied = True

    if not augmentation_applied:
        augmented = add_minimal_noise(augmented)

    return augmented


def process_images():
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    
    enhance_ratio = 0.15  # For example, 25% of images will have one augmented copy with test enhancement
    num_enhanced = int(total_images * enhance_ratio)
    
    output_dir = os.path.join(IMAGE_DIR, "augmented")
    os.makedirs(output_dir, exist_ok=True)

    for i, filename in enumerate(tqdm(image_files, desc="Processing images")):
        img_path = os.path.join(IMAGE_DIR, filename)
        # Read the image in color mode (default)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {filename}")
            continue
        
        original_name = f"{os.path.splitext(filename)[0]}_original.jpeg"
        cv2.imwrite(os.path.join(output_dir, original_name), img)

        for j in range(3):
            # Apply the test enhancement to a subset of images (e.g., first 25%) for one of the augmentations
            apply_test_flag = (i < num_enhanced) and (j == 0)
            augmented = augment_image(img, apply_test_enhancement=apply_test_flag)
            
            aug_name = f"{os.path.splitext(filename)[0]}_aug{j+1}.jpg"
            cv2.imwrite(os.path.join(output_dir, aug_name), augmented)

if __name__ == "__main__":
    process_images()
