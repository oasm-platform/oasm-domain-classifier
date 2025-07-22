import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel

class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config, **kwargs):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(
            config["base_model"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # Disable gradients for inference
            features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            dropped = self.dropout(features)
            outputs = self.fc(dropped)
            return torch.softmax(outputs[:, 0, :], dim=1)
