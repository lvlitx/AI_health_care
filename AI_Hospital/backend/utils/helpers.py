import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from typing import List, Tuple, Union, Dict
import os
import re
from collections import Counter
import torchvision.transforms as transforms
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

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
