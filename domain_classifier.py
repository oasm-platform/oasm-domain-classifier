import torch
from transformers import AutoTokenizer, AutoConfig
from model import Model

class DomainClassifier:
    def __init__(self, model_name="nvidia/multilingual-domain-classifier"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = Model.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        if not text.strip():
            return {"error": "Empty input text."}

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )

        if inputs['input_ids'].size(1) == 0:
            return {"error": "Empty input after tokenization."}

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(inputs["input_ids"], inputs["attention_mask"])

            if outputs.dim() == 1:
                probs = outputs.unsqueeze(0)
            elif outputs.dim() == 2:
                probs = outputs
            else:
                return {"error": f"Unexpected output shape: {outputs.shape}"}

        batch_probs = probs[0]
        predicted_class_idx = torch.argmax(batch_probs).item()
        predicted_domain = self.config.id2label[predicted_class_idx]

        results = {
            self.config.id2label[i]: float(batch_probs[i])
            for i in range(len(batch_probs))
        }

        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

        return {
            "predicted_domain": predicted_domain,
            "confidence": results[predicted_domain],
            "all_probabilities": sorted_results
        }
