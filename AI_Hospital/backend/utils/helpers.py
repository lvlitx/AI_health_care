import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from typing import List, Tuple, Union, Dict, Optional
import os
import re
from collections import Counter
import torchvision.transforms as transforms
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pytesseract
from PIL import Image as PILImage
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

def preprocess_text(text: str) -> str:
    """
    Preprocess text by converting to lowercase, removing special characters,
    and tokenizing.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    text = ' '.join(text.split())
    
    return text

def get_text_embeddings(text: str, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> np.ndarray:
    """
    Get text embeddings using a transformer model.
    
    Args:
        text (str): Input text
        model_name (str): Name of the transformer model to use
        
    Returns:
        np.ndarray: Text embeddings
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use CLS token embedding
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (height, width)
        
    Returns:
        np.ndarray: Preprocessed image
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Normalize pixel values
    image = image.astype(np.float32) / 255.0
    
    return image

def extract_features(image: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    """
    Extract features from an image using a pre-trained model.
    
    Args:
        image (np.ndarray): Preprocessed image
        model (torch.nn.Module): Pre-trained model
        
    Returns:
        np.ndarray: Image features
    """
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    image_tensor = image_tensor.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
    
    with torch.no_grad():
        features = model(image_tensor)
    
    return features.numpy()

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1 (np.ndarray): First vector
        vec2 (np.ndarray): Second vector
        
    Returns:
        float: Cosine similarity score
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return dot_product / (norm1 * norm2)

def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with stopwords removed
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save image to disk.
    
    Args:
        image (np.ndarray): Image to save
        output_path (str): Path where to save the image
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to PIL Image and save
    image = Image.fromarray((image * 255).astype(np.uint8))
    image.save(output_path)

def load_image(image_path: str) -> np.ndarray:
    """
    Load image from disk.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Loaded image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_text_summary(text: str, num_sentences: int = 3) -> str:
    """
    Generate a summary of text using TF-IDF scoring.
    
    Args:
        text (str): Input text
        num_sentences (int): Number of sentences in summary
        
    Returns:
        str: Text summary
    """
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calculate sentence scores
    sentence_scores = tfidf_matrix.sum(axis=1).A1
    
    # Get top sentences
    top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
    summary = ' '.join([sentences[i] for i in sorted(top_indices)])
    
    return summary

def extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    """
    Extract keywords from text using TF-IDF.
    
    Args:
        text (str): Input text
        num_keywords (int): Number of keywords to extract
        
    Returns:
        List[str]: List of keywords
    """
    vectorizer = TfidfVectorizer(max_features=num_keywords)
    tfidf_matrix = vectorizer.fit_transform([text])
    
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    # Get top keywords
    top_indices = tfidf_scores.argsort()[-num_keywords:][::-1]
    keywords = [feature_names[i] for i in top_indices]
    
    return keywords

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Detected language code
    """
    from langdetect import detect
    try:
        return detect(text)
    except:
        return "unknown"

def apply_image_augmentation(image: np.ndarray, 
                           brightness_factor: float = 1.2,
                           contrast_factor: float = 1.2) -> np.ndarray:
    """
    Apply basic image augmentation.
    
    Args:
        image (np.ndarray): Input image
        brightness_factor (float): Brightness adjustment factor
        contrast_factor (float): Contrast adjustment factor
        
    Returns:
        np.ndarray: Augmented image
    """
    # Convert to PIL Image
    image_pil = Image.fromarray(image)
    
    # Apply brightness and contrast
    enhancer = transforms.ColorJitter(brightness=brightness_factor, 
                                    contrast=contrast_factor)
    augmented_image = enhancer(image_pil)
    
    return np.array(augmented_image)

def extract_face(image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Detect and extract faces from an image.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        Tuple[np.ndarray, List[Tuple[int, int, int, int]]]: 
            Processed image and list of face bounding boxes
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return image, faces

def plot_image_with_boxes(image: np.ndarray, 
                         boxes: List[Tuple[int, int, int, int]], 
                         labels: List[str] = None) -> None:
    """
    Plot image with bounding boxes.
    
    Args:
        image (np.ndarray): Input image
        boxes (List[Tuple[int, int, int, int]]): List of bounding boxes
        labels (List[str], optional): Labels for each box
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    for i, (x, y, w, h) in enumerate(boxes):
        rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
        plt.gca().add_patch(rect)
        if labels and i < len(labels):
            plt.text(x, y-10, labels[i], color='red')
    
    plt.axis('off')
    plt.show()

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using cosine similarity.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Similarity score
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def get_text_statistics(text: str) -> Dict:
    """
    Calculate various text statistics.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Dictionary containing text statistics
    """
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    
    stats = {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': np.mean([len(word) for word in words]),
        'avg_sentence_length': np.mean([len(word_tokenize(sent)) for sent in sentences]),
        'unique_words': len(set(words)),
        'word_frequency': dict(Counter(words).most_common(10))
    }
    
    return stats

def apply_image_filter(image: np.ndarray, 
                      filter_type: str = 'blur',
                      kernel_size: int = 3) -> np.ndarray:
    """
    Apply various image filters.
    
    Args:
        image (np.ndarray): Input image
        filter_type (str): Type of filter ('blur', 'sharpen', 'edge')
        kernel_size (int): Kernel size for the filter
        
    Returns:
        np.ndarray: Filtered image
    """
    if filter_type == 'blur':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == 'sharpen':
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == 'edge':
        return cv2.Canny(image, 100, 200)
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

def visualize_attention(image: np.ndarray, 
                       attention_map: np.ndarray,
                       alpha: float = 0.5) -> np.ndarray:
    """
    Visualize attention map on the image.
    
    Args:
        image (np.ndarray): Input image
        attention_map (np.ndarray): Attention map
        alpha (float): Transparency factor
        
    Returns:
        np.ndarray: Image with attention visualization
    """
    # Normalize attention map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    
    # Resize attention map to match image size
    attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    
    # Blend with original image
    output = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
    
    return output

def load_object_detection_model(num_classes: int = 91) -> torch.nn.Module:
    """
    Load a pre-trained Faster R-CNN model for object detection.
    
    Args:
        num_classes (int): Number of classes to detect
        
    Returns:
        torch.nn.Module: Loaded model
    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def perform_object_detection(image: np.ndarray, 
                           model: torch.nn.Module,
                           confidence_threshold: float = 0.5) -> Tuple[List[Dict], np.ndarray]:
    """
    Perform object detection on an image.
    
    Args:
        image (np.ndarray): Input image
        model (torch.nn.Module): Object detection model
        confidence_threshold (float): Confidence threshold for detections
        
    Returns:
        Tuple[List[Dict], np.ndarray]: Detections and annotated image
    """
    # Preprocess image
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Process predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    
    # Filter by confidence
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Draw boxes
    image_with_boxes = image.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_with_boxes, f'Class {label}: {score:.2f}', 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    detections = [{'box': box, 'score': score, 'label': label} 
                 for box, score, label in zip(boxes, scores, labels)]
    
    return detections, image_with_boxes

def perform_ocr(image: np.ndarray, 
                lang: str = 'eng',
                config: str = '--psm 6') -> str:
    """
    Perform OCR on an image.
    
    Args:
        image (np.ndarray): Input image
        lang (str): Language for OCR
        config (str): Tesseract configuration
        
    Returns:
        str: Extracted text
    """
    # Convert numpy array to PIL Image
    pil_image = PILImage.fromarray(image)
    
    # Perform OCR
    text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
    return text.strip()

def prepare_transfer_learning_model(base_model_name: str = 'resnet50',
                                  num_classes: int = 10) -> torch.nn.Module:
    """
    Prepare a model for transfer learning.
    
    Args:
        base_model_name (str): Name of the base model
        num_classes (int): Number of classes for the new task
        
    Returns:
        torch.nn.Module: Prepared model
    """
    # Load pre-trained model
    if base_model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif base_model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {base_model_name}")
    
    return model

class CustomDataset(Dataset):
    """
    Custom dataset class for transfer learning.
    """
    def __init__(self, images: List[np.ndarray], labels: List[int], transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_transfer_learning_model(model: torch.nn.Module,
                                train_loader: DataLoader,
                                num_epochs: int = 10,
                                learning_rate: float = 0.001) -> List[float]:
    """
    Train a transfer learning model.
    
    Args:
        model (torch.nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        num_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        
    Returns:
        List[float]: Training losses
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
    return losses

def extract_text_regions(image: np.ndarray) -> List[np.ndarray]:
    """
    Extract regions containing text from an image.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        List[np.ndarray]: List of text regions
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and extract text regions
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:  # Filter small regions
            region = image[y:y+h, x:x+w]
            text_regions.append(region)
    
    return text_regions

def visualize_detections(image: np.ndarray,
                        detections: List[Dict],
                        class_names: List[str] = None) -> np.ndarray:
    """
    Visualize object detections on an image.
    
    Args:
        image (np.ndarray): Input image
        detections (List[Dict]): List of detections
        class_names (List[str]): List of class names
        
    Returns:
        np.ndarray: Image with visualizations
    """
    image_with_dets = image.copy()
    
    for det in detections:
        box = det['box']
        score = det['score']
        label = det['label']
        
        x1, y1, x2, y2 = box.astype(int)
        class_name = class_names[label] if class_names else f'Class {label}'
        
        # Draw box
        cv2.rectangle(image_with_dets, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label_text = f'{class_name}: {score:.2f}'
        cv2.putText(image_with_dets, label_text, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image_with_dets

class OCRDataset(Dataset):
    """
    Custom dataset for OCR training.
    """
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[str],
                 transform: Optional[transforms.Compose] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for OCR.
    """
    def __init__(self, 
                 num_classes: int,
                 hidden_size: int = 256,
                 num_layers: int = 2):
        super(CRNN, self).__init__()
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # RNN layers
        self.rnn = nn.LSTM(
            input_size=512 * 2,  # Adjust based on your input image size
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # CNN
        conv = self.cnn(x)
        batch, channel, height, width = conv.size()
        
        # Reshape for RNN
        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(0, 2, 1)
        
        # RNN
        rnn, _ = self.rnn(conv)
        
        # Output
        output = self.fc(rnn)
        return output

def train_ocr_model(model: nn.Module,
                    train_loader: DataLoader,
                    num_epochs: int = 10,
                    learning_rate: float = 0.001,
                    device: str = 'cuda') -> Dict[str, List[float]]:
    """
    Train the OCR model.
    
    Args:
        model (nn.Module): OCR model
        train_loader (DataLoader): Training data loader
        num_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        device (str): Device to train on
        
    Returns:
        Dict[str, List[float]]: Training history
    """
    model = model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return history

def get_ocr_transforms(train: bool = True) -> transforms.Compose:
    """
    Get data transforms for OCR training/inference.
    
    Args:
        train (bool): Whether to use training transforms
        
    Returns:
        transforms.Compose: Image transforms
    """
    if train:
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.ElasticTransform(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return transform

def predict_text(model: nn.Module,
                image: np.ndarray,
                device: str = 'cuda') -> str:
    """
    Predict text from image using trained OCR model.
    
    Args:
        model (nn.Module): Trained OCR model
        image (np.ndarray): Input image
        device (str): Device to run inference on
        
    Returns:
        str: Predicted text
    """
    model.eval()
    transform = get_ocr_transforms(train=False)
    
    # Preprocess image
    image = transform(image=image)['image']
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
    # Convert predictions to text
    # You'll need to implement this based on your character mapping
    text = decode_predictions(predicted)
    
    return text

def decode_predictions(predictions: torch.Tensor,
                      char_map: Dict[int, str]) -> str:
    """
    Decode model predictions to text.
    
    Args:
        predictions (torch.Tensor): Model predictions
        char_map (Dict[int, str]): Character mapping dictionary
        
    Returns:
        str: Decoded text
    """
    text = ''
    prev_char = None
    
    for pred in predictions:
        char = char_map[pred.item()]
        if char != prev_char and char != '<blank>':
            text += char
        prev_char = char
    
    return text

def create_char_map(vocabulary: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create character mapping dictionaries.
    
    Args:
        vocabulary (str): String containing all possible characters
        
    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: Character to index and index to character mappings
    """
    char_to_idx = {'<blank>': 0}
    idx_to_char = {0: '<blank>'}
    
    for i, char in enumerate(vocabulary, 1):
        char_to_idx[char] = i
        idx_to_char[i] = char
    
    return char_to_idx, idx_to_char

def prepare_ocr_data(image_paths: List[str],
                     labels: List[str],
                     char_map: Dict[str, int],
                     batch_size: int = 32) -> DataLoader:
    """
    Prepare data for OCR training.
    
    Args:
        image_paths (List[str]): List of image paths
        labels (List[str]): List of text labels
        char_map (Dict[str, int]): Character mapping dictionary
        batch_size (int): Batch size
        
    Returns:
        DataLoader: Data loader for training
    """
    transform = get_ocr_transforms(train=True)
    dataset = OCRDataset(image_paths, labels, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
