import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Union

class ClinicalBERTModel:
    def __init__(self):
        """Initialize the ClinicalBERT model and tokenizer"""
        self.model_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text using ClinicalBERT
        
        Args:
            text: Single string or list of strings to encode
            
        Returns:
            torch.Tensor: Encoded text embeddings
        """
        # Prepare inputs
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the [CLS] token embeddings as the sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return embeddings

    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            float: Cosine similarity score
        """
        # Encode both texts
        embedding1 = self.encode_text(text1)
        embedding2 = self.encode_text(text2)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        
        return similarity.item()

    def batch_process(self, texts: List[str]) -> torch.Tensor:
        """
        Process a batch of texts to get their embeddings
        
        Args:
            texts: List of text strings to process
            
        Returns:
            torch.Tensor: Batch of embeddings
        """
        return self.encode_text(texts)
