import argparse
import torch
from classifier import DomainClassifier

def main():
    parser = argparse.ArgumentParser(description='Domain Classification Tool')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    try:
        print("Loading model components...")
        classifier = DomainClassifier(
            model_name="nvidia/domain-classifier",
            use_cache=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\nInteractive Domain Classification")
    print("Enter texts to classify (type 'quit' to exit):")
    print("-" * 50)
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if not text:
                continue
                
            result = classifier.predict_single(text)
            print(f"Predicted domain: {result}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Enable optimizations
    torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False
    torch.set_num_threads(4)  # Optimize CPU usage
    main()