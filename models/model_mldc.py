import torch
from torch import nn
from transformers import AutoModel
from huggingface_hub import PyTorchModelHubMixin


class ModelMldc(nn.Module, PyTorchModelHubMixin):
    """
    Custom model class required for NVIDIA's multilingual domain classifier
    
    This class implements the specific architecture needed for the pre-trained model
    from nvidia/multilingual-domain-classifier on HuggingFace.
    """
    
    def __init__(self, config):
        """
        Initialize the model
        
        Args:
            config: Model configuration containing base_model, fc_dropout, and id2label
        """
        super(ModelMldc, self).__init__()
        
        # Load the base transformer model
        self.model = AutoModel.from_pretrained(config["base_model"])
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(config["fc_dropout"])
        
        # Final classification layer
        self.fc = nn.Linear(
            self.model.config.hidden_size, 
            len(config["id2label"])
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
            
        Returns:
            torch.Tensor: Softmax probabilities for each class
        """
        # Get features from the base model
        features = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).last_hidden_state
        
        # Apply dropout
        dropped = self.dropout(features)
        
        # Get logits from classification layer
        outputs = self.fc(dropped)
        
        # Apply softmax to get probabilities
        # Note: Uses [:, 0, :] to get the [CLS] token representation
        return torch.softmax(outputs[:, 0, :], dim=1)