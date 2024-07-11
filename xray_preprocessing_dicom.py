import argparse
import pydicom
import numpy as np
import cv2

def apply_gaussian_blur(image, ksize, sigma):
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def modify_dicom_image(dicom_path, output_path, blur_ksize=5, blur_sigma=1, apply_sharp=True, apply_clahe=True):
    # Load DICOM file
    dicom = pydicom.dcmread(dicom_path)

    # Convert DICOM pixel data to numpy array
    image = dicom.pixel_array

    # Normalize the image to 8-bit (0-255) for OpenCV processing
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Gaussian blur
    blurred_image = apply_gaussian_blur(image, blur_ksize, blur_sigma)
    
    # Apply sharpening if needed
    if apply_sharp:
        sharpened_image = apply_sharpening(blurred_image)
    else:
        sharpened_image = blurred_image

    # Apply CLAHE if needed
    if apply_clahe:
        clahe_image = apply_clahe(sharpened_image)
    else:
        clahe_image = sharpened_image

    # Convert the processed image back to original bit depth
    final_image = cv2.normalize(clahe_image, None, 0, np.iinfo(dicom.pixel_array.dtype).max, cv2.NORM_MINMAX).astype(dicom.pixel_array.dtype)

    # Update the DICOM file with the modified pixel data
    modified_dicom = dicom.copy()
    modified_dicom.PixelData = final_image.tobytes()

    # Save the modified DICOM file
    modified_dicom.save_as(output_path)

    print(f"Modified DICOM image saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modify DICOM image with Gaussian blur, sharpening, and CLAHE.')
    parser.add_argument('dicom_path', type=str, help='Path to the input DICOM file')
    parser.add_argument('output_path', type=str, help='Path to save the output DICOM file')
    parser.add_argument('--blur_ksize', type=int, default=5, help='Kernel size for Gaussian blur')
    parser.add_argument('--blur_sigma', type=float, default=1, help='Sigma for Gaussian blur')
    parser.add_argument('--apply_sharp', action='store_true', help='Apply sharpening')
    parser.add_argument('--apply_clahe', action='store_true', help='Apply CLAHE')

    args = parser.parse_args()

    modify_dicom_image(
        args.dicom_path, 
        args.output_path, 
        blur_ksize=args.blur_ksize, 
        blur_sigma=args.blur_sigma, 
        apply_sharp=args.apply_sharp, 
        apply_clahe=args.apply_clahe
    )
