import os
import re
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoConfig
from model import CustomModel

class DomainClassifier:
    def __init__(self, model_name: str = "nvidia/domain-classifier", use_cache: bool = True):
        """
        Initialize the domain classifier with optimizations
        
        Args:
            model_name: HuggingFace model name
            use_cache: Whether to use cached models (faster loading)
        """
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
            use_fast=True,
            local_files_only=use_cache and os.path.exists('./model_cache')
        )
        
        self.model = CustomModel.from_pretrained(
            model_name, 
            timeout=60,
            local_files_only=use_cache and os.path.exists('./model_cache')
        )
        
        self.model.eval()
        
        # Set up requests session with headers to mimic a browser
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Enable torch.jit optimization if possible
        try:
            self.model = torch.jit.optimize_for_inference(self.model)
        except:
            pass
        
        print(f"Model loaded successfully! Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text from HTML"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        return text.strip()

    def extract_text_from_url(self, url: str) -> str:
        """Extract and clean text from a given URL"""
        try:
            # Add https:// if no scheme is present
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML and extract text
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text and clean it
            text = soup.get_text()
            return self._clean_text(text)
            
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return ""

    def predict_domain_from_url(self, url: str) -> str:
        """Predict domain category from a website URL"""
        print(f"\nFetching content from: {url}")
        text = self.extract_text_from_url(url)
        if not text:
            return "Could not fetch or extract meaningful content from the URL"
            
        print("Classifying content...")
        return self.predict_single(text)

    def predict(self, texts: list[str], batch_size: int = 32) -> list[str]:
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
                max_length=512 
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
