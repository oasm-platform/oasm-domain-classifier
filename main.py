import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import PyTorchModelHubMixin
import argparse
from typing import List
import os

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

class DomainClassifier:
    def __init__(self, model_name: str = "nvidia/domain-classifier", use_cache: bool = True):
        """
        Initialize the domain classifier with optimizations
        
        Args:
            model_name: HuggingFace model name
            use_cache: Whether to use cached models (faster loading)
        """
        print("Loading model components...")
        
        # Set cache directory if specified
        if use_cache:
            os.environ['TRANSFORMERS_CACHE'] = './model_cache'
        
        # Load components with optimizations
        self.config = AutoConfig.from_pretrained(
            model_name, 
            timeout=60,
            local_files_only=use_cache and os.path.exists('./model_cache')
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            timeout=60,
            use_fast=True,  # Use fast tokenizer for better performance
            local_files_only=use_cache and os.path.exists('./model_cache')
        )
        
        self.model = CustomModel.from_pretrained(
            model_name, 
            timeout=60,
            local_files_only=use_cache and os.path.exists('./model_cache')
        )
        
        self.model.eval()
        
        # Enable torch.jit optimization if possible
        try:
            self.model = torch.jit.optimize_for_inference(self.model)
        except:
            pass  # Continue without JIT optimization if not supported
        
        print(f"Model loaded successfully! Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    def predict(self, texts: List[str], batch_size: int = 32) -> List[str]:
        """
        Predict domains for given texts with batch processing
        
        Args:
            texts: List of text samples to classify
            batch_size: Number of samples to process at once
            
        Returns:
            List of predicted domain labels
        """
        all_predictions = []
        
        # Process in batches for better memory efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize with optimizations
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=512  # Limit max length for faster processing
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(inputs["input_ids"], inputs["attention_mask"])
                predicted_classes = torch.argmax(outputs, dim=1)
                
                # Convert to labels
                batch_predictions = [
                    self.config.id2label[class_idx.item()] 
                    for class_idx in predicted_classes.cpu()
                ]
                all_predictions.extend(batch_predictions)
        
        return all_predictions

    def predict_single(self, text: str) -> str:
        """
        Predict domain for a single text (optimized for single predictions)
        """
        return self.predict([text])[0]

def main():
    parser = argparse.ArgumentParser(description='Domain Classification Tool')
    parser.add_argument('--text', type=str, help='Single text to classify')
    parser.add_argument('--file', type=str, help='File containing texts (one per line)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--model', type=str, default='nvidia/domain-classifier', 
                       help='Model name from HuggingFace')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for processing multiple texts')
    parser.add_argument('--no-cache', action='store_true', 
                       help='Disable model caching')
    
    args = parser.parse_args()
    
    # Initialize classifier
    try:
        classifier = DomainClassifier(
            model_name=args.model,
            use_cache=not args.no_cache
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Handle different input modes
    if args.text:
        # Single text classification
        result = classifier.predict_single(args.text)
        print(f"Text: {args.text}")
        print(f"Predicted domain: {result}")
        
    elif args.file:
        # File-based classification
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
            
            print(f"Processing {len(texts)} texts from {args.file}...")
            results = classifier.predict(texts, batch_size=args.batch_size)
            
            for text, domain in zip(texts, results):
                print(f"Text: {text}")
                print(f"Domain: {domain}")
                print("-" * 50)
                
        except FileNotFoundError:
            print(f"File {args.file} not found!")
            return
        except Exception as e:
            print(f"Error processing file: {e}")
            return
    
    elif args.interactive:
        # Interactive mode
        print("Interactive Domain Classification")
        print("Enter texts to classify (type 'quit' to exit):")
        print("-" * 50)
        
        while True:
            try:
                text = input("Enter text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if not text:
                    continue
                
                result = classifier.predict_single(text)
                print(f"Predicted domain: {result}")
                print("-" * 30)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    else:
        # Default: Demo with sample texts
        print("Running demo with sample texts...")
        text_samples = [
            "Sports is a popular domain",
            "Politics is a popular domain",
            "The latest technology trends are fascinating",
            "Breaking news from around the world"
        ]
        
        results = classifier.predict(text_samples)
        
        print("Results:")
        print("-" * 50)
        for text, domain in zip(text_samples, results):
            print(f"Text: {text}")
            print(f"Domain: {domain}")
            print("-" * 30)

if __name__ == "__main__":
    # Enable optimizations
    torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False
    torch.set_num_threads(4)  # Optimize CPU usage
    
    main()