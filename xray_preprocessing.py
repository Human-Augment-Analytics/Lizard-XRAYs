import argparse
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from PIL import Image, ImageEnhance

def dcm_to_jpeg(dcm_file_path, jpeg_file_path):
    dicom = pydicom.dcmread(dcm_file_path)
    
    # If DICOM file has a VOI LUT (Value of Interest Look-Up Table), apply it
    if hasattr(dicom, 'VOILUTFunction'):
        image = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        image = dicom.pixel_array
    
    image = image - np.min(image)
    image = image / np.max(image)
    image = (image * 255).astype(np.uint8)

    # Convert to RGB
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image = np.stack((image,) * 3, axis=-1)
    
    pil_image = Image.fromarray(image)
    pil_image.save(jpeg_file_path, 'JPEG')

# Function to enhance image sharpness, contrast and apply Gaussian blur
def enhance_image(image_path, sharpness=4, contrast=1.3, blur=3):
    """Enhance image sharpness, contrast, and blur.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the enhanced image.
        sharpness (float, optional): Sharpness level. Defaults to 4.
        contrast (float, optional): Contrast level. Defaults to 1.3.
        blur (int, optional): Blur level. Defaults to 3.
    """

    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to PIL Image
    pil_img = Image.fromarray(img)

    # Enhance the sharpness
    enhancer = ImageEnhance.Sharpness(pil_img)
    img_enhanced = enhancer.enhance(sharpness)

    # Enhance the contrast
    enhancer = ImageEnhance.Contrast(img_enhanced)
    img_enhanced = enhancer.enhance(contrast)

    # Convert back to OpenCV image (numpy array)
    img_enhanced = np.array(img_enhanced)

    # Apply a small amount of Gaussian blur
    img_enhanced = cv2.GaussianBlur(img_enhanced, (blur, blur), 0)

    # Convert back to PIL Image and save
    img_enhanced = Image.fromarray(img_enhanced)

    return img_enhanced

def clahe(image, clipLimit=2.0, tileGridSize=(8, 8)):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def image_complement(image):
    img_complement = cv2.bitwise_not(image)
    return img_complement

def gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply image enhancement techniques to improve XRAY image.')
    parser.add_argument('input_path', type=str, help='Path to the input image')
    parser.add_argument('output_path', type=str, help='Path to store the output image')
    parser.add_argument('--sharpness', type=float, default=4.0, help='Sharpness level')
    parser.add_argument('--contrast', type=float, default=1.3, help='Contrast level')
    parser.add_argument('--blur', type=int, default=3, help='Blur for Gaussian Blur')
    parser.add_argument('--clip_limit', type=float, default=2.0, help='Clip limit for CLAHE')
    parser.add_argument('--tile_grid_size', type=int, nargs=2, default=(8, 8), help='Tile grid size for CLAHE (two integers)')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma for gamma correction')

    args = parser.parse_args()

    image = cv2.imread(args.input_path)

    image = clahe(image, args.clip_limit, args.tile_grid_size)
    image = gamma_correction(image, args.gamma)
    image = enhance_image(image)

    cv2.imwrite(args.output_path, image)