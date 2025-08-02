import torch
from transformers import AutoTokenizer, AutoConfig
from model import Model
from crawl_web import CrawlWeb
import json


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


def main():
    url = "https://dichvucong.gov.vn/p/home/dvc-trang-chu.html"

    classifier = DomainClassifier()
    crawler = CrawlWeb()
    text = crawler.crawl(url)

    print(f"Crawled text: {text}")

    result = classifier.predict(text)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"Predicted domain: {result['predicted_domain']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nAll probabilities:")
    print(json.dumps(result['all_probabilities'], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
