import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, Optional
from .ocr import OCRModel

class OCRFineTuner:
    def __init__(self, base_model: Optional[OCRModel] = None):
        """
        Initialize OCR model fine-tuning.
        
        Args:
            base_model: Pre-trained OCR model to fine-tune, creates new one if None
        """
        try:
            self.model = base_model if base_model else OCRModel()
            self.device = self.model.device
            logging.info(f"OCR fine-tuning initialized on {self.device}")
        except Exception as e:
            logging.error(f"Error initializing fine-tuning: {str(e)}")
            raise

    def finetune(self, 
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 epochs: int = 10,
                 learning_rate: float = 0.0001,
                 early_stopping_patience: int = 5) -> Dict[str, list]:
        """
        Fine-tune the OCR model on custom dataset.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
            early_stopping_patience: Number of epochs to wait before early stopping
            
        Returns:
            dict: Training history with losses and metrics
        """
        try:
            # Initialize optimizer with lower learning rate for fine-tuning
            optimizer = optim.Adam(self.model.model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            history = {
                'train_loss': [],
                'val_loss': [] if val_loader else None
            }
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training phase
                self.model.model.train()
                train_loss = 0.0
                
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model.model(images)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                avg_train_loss = train_loss / len(train_loader)
                history['train_loss'].append(avg_train_loss)
                
                # Validation phase
                if val_loader:
                    val_loss = self._validate(val_loader, criterion)
                    history['val_loss'].append(val_loss)
                    
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= early_stopping_patience:
                        logging.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
                        
                logging.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Train Loss: {avg_train_loss:.4f}"
                           f"{f' - Val Loss: {val_loss:.4f}' if val_loader else ''}")
                
            return history
            
        except Exception as e:
            logging.error(f"Error during fine-tuning: {str(e)}")
            raise
            
    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """
        Perform validation step.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            float: Validation loss
        """
        self.model.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model.model(images)
                val_loss += criterion(outputs, labels).item()
                
        return val_loss / len(val_loader)

if __name__ == "__main__":
    # Example usage
    try:
        # Initialize fine-tuner with pre-trained model
        fine_tuner = OCRFineTuner()
        
        # Assuming you have prepared your data loaders
        # train_loader = DataLoader(...)
        # val_loader = DataLoader(...)
        
        # Fine-tune the model
        # history = fine_tuner.finetune(train_loader, val_loader)
        # print("Training history:", history)
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
