import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import logging
from typing import List, Union, Dict
import numpy as np

class OCRModel:
    def __init__(self, num_classes: int = 62, pretrained: bool = True):
        """
        Initialize OCR model using transfer learning with ResNet50 backbone.
        
        Args:
            num_classes (int): Number of character classes (default 62 for 0-9, a-z, A-Z)
            pretrained (bool): Whether to use pretrained weights
        """
        try:
            # Initialize ResNet50 backbone with pretrained weights
            self.model = models.resnet50(pretrained=pretrained)
            
            # Modify final fully connected layer for OCR task
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            # Define image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Initialize training components
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            logging.info(f"OCR model initialized successfully on {self.device}")
            
        except Exception as e:
            logging.error(f"Error initializing OCR model: {str(e)}")
            raise
            
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Perform one training step.
        
        Args:
            images: Batch of input images
            labels: Corresponding character labels
            
        Returns:
            float: Loss value
        """
        try:
            self.model.train()
            self.optimizer.zero_grad()
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            logging.error(f"Error in training step: {str(e)}")
            raise
            
    def predict(self, image: Union[str, Image.Image]) -> Dict[str, Union[str, float]]:
        """
        Predict character from input image.
        
        Args:
            image: Input image path or PIL Image object
            
        Returns:
            dict: Dictionary containing predicted character and confidence score
        """
        try:
            self.model.eval()
            
            # Load and preprocess image
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                pred_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_idx].item()
                
                # Convert index to character
                if pred_idx < 10:
                    char = str(pred_idx)
                elif pred_idx < 36:
                    char = chr(pred_idx - 10 + ord('A'))
                else:
                    char = chr(pred_idx - 36 + ord('a'))
                
                return {
                    "predicted_char": char,
                    "confidence": confidence
                }
                
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise
            
    def save_model(self, path: str):
        """
        Save model weights to specified path.
        
        Args:
            path (str): Path to save model weights
        """
        try:
            torch.save(self.model.state_dict(), path)
            logging.info(f"Model saved successfully to {path}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, path: str):
        """
        Load model weights from specified path.
        
        Args:
            path (str): Path to load model weights from
        """
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logging.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

if __name__ == "__main__":
    #  usage
    try:
        ocr_model = OCRModel()
        
        # Example training step (assuming you have training data)
        # train_images = torch.randn(32, 3, 224, 224)
        # train_labels = torch.randint(0, 62, (32,))
        # loss = ocr_model.train_step(train_images, train_labels)
        # print(f"Training loss: {loss}")
        
        # Example prediction
        # result = ocr_model.predict("path_to_image.jpg")
        # print(f"Predicted character: {result['predicted_char']}")
        # print(f"Confidence: {result['confidence']}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
