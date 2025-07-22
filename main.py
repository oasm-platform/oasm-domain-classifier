#!/usr/bin/env python3
"""
Domain Classifier - A tool for classifying text domains using pre-trained models.
"""
import argparse
import torch
from classifier import DomainClassifier

def main():
    url = "24h.com.vn"
    
    try:
        print("Loading model components...")
        classifier = DomainClassifier(
            model_name="nvidia/domain-classifier",
            use_cache=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    result = classifier.predict_domain_from_url(url)
    print(f"\nPredicted domain category: {result}")

        

if __name__ == "__main__":
    # Enable optimizations
    torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False
    torch.set_num_threads(4)  # Optimize CPU usage
    main()