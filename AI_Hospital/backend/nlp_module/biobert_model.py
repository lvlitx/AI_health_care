import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Union, Dict
import logging

class BioBERTModel:
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.2"):
        """
        Initialize the BioBERT model with the specified pre-trained model.
        
        Args:
            model_name (str): The name of the pre-trained BioBERT model to use
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"BioBERT model loaded successfully on {self.device}")
        except Exception as e:
            logging.error(f"Error loading BioBERT model: {str(e)}")
            raise

    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for the input text(s) using BioBERT.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy.ndarray: Text embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            # Tokenize texts
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move inputs to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            # Use [CLS] token embeddings as sentence embeddings
            embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            return embeddings
            
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise

    def process_medical_text(self, text: str) -> Dict[str, float]:
        """
        Process medical text to extract relevant biomedical information and similarity scores.
        
        Args:
            text (str): Input medical text to process
            
        Returns:
            dict: Dictionary containing processed information and scores
        """
        try:
            # Get text embeddings
            embeddings = self.get_embeddings(text)
            
            # Calculate basic statistics
            embedding_norm = float(np.linalg.norm(embeddings))
            embedding_mean = float(np.mean(embeddings))
            embedding_std = float(np.std(embeddings))
            
            return {
                "embedding_norm": embedding_norm,
                "embedding_mean": embedding_mean,
                "embedding_std": embedding_std,
                "embedding_dimension": embeddings.shape[1]
            }
            
        except Exception as e:
            logging.error(f"Error processing medical text: {str(e)}")
            raise

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the cosine similarity between two medical texts.
        
        Args:
            text1 (str): First medical text
            text2 (str): Second medical text
            
        Returns:
            float: Cosine similarity score between the two texts
        """
        try:
            # Get embeddings for both texts
            embedding1 = self.get_embeddings(text1)
            embedding2 = self.get_embeddings(text2)
            
            # Calculate cosine similarity
            similarity = float(
                np.dot(embedding1.flatten(), embedding2.flatten()) /
                (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            )
            
            return similarity
            
        except Exception as e:
            logging.error(f"Error calculating similarity: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    try:
        biobert = BioBERTModel()
        
        # Test text processing
        sample_text = "Patient presents with acute myocardial infarction and elevated troponin levels."
        result = biobert.process_medical_text(sample_text)
        print("Processing Result:", result)
        
        # Test similarity calculation
        text1 = "Patient has diabetes mellitus type 2"
        text2 = "Patient diagnosed with type 2 diabetes"
        similarity = biobert.calculate_similarity(text1, text2)
        print(f"Similarity score: {similarity}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
