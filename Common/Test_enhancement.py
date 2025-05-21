import os
import cv2
import numpy as np

input_folder = r"Path"
output_folder = r"Path"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, strength=2):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened

def apply_clahe_color(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def test_enhancement(image):
    clahe_enhanced = apply_clahe_color(image)
    gamma_corrected = adjust_gamma(clahe_enhanced, gamma=0.8)
    final_transformed = unsharp_mask(gamma_corrected, kernel_size=(5, 5), sigma=1.0, strength=2)
    return final_transformed

if __name__ == "__main__":
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        image = cv2.imread(input_path)

        if image is None:
            print(f"Chyba při načítání: {input_path}")
            continue

        enhanced_image = test_enhancement(image)

        cv2.imwrite(output_path, enhanced_image)

    print("Done")
