import cv2
import numpy as np
from PIL import Image
import logging
from typing import Union, Tuple

class ImagePreprocessor:
    def __init__(self):
        """Initialize the image preprocessor"""
        logging.info("Initializing image preprocessor")

    def preprocess(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for OCR by applying various enhancement techniques.
        
        Args:
            image: Input image as file path, numpy array, or PIL Image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Load and standardize input
            if isinstance(image, str):
                image = cv2.imread(image)
            elif isinstance(image, Image.Image):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 
                2
            )
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(thresh)
            
            # Deskew if needed
            angle = self._get_skew_angle(denoised)
            if abs(angle) > 0.5:
                denoised = self._rotate_image(denoised, angle)
                
            return denoised
            
        except Exception as e:
            logging.error(f"Error in image preprocessing: {str(e)}")
            raise
            
    def _get_skew_angle(self, image: np.ndarray) -> float:
        """
        Calculate skew angle of text in image.
        
        Args:
            image: Input image
            
        Returns:
            float: Skew angle in degrees
        """
        # Find all non-zero points
        coords = np.column_stack(np.where(image > 0))
        
        # Get angle
        angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
        
        # Adjust angle
        if angle < -45:
            angle = 90 + angle
            
        return -angle
        
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by given angle.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            numpy.ndarray: Rotated image
        """
        # Get image dimensions
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
