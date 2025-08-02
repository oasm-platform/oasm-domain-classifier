# OASM Domain Classifier

A Python package for classifying web content into different domains using NVIDIA's multilingual domain classification model. This tool can analyze both direct text input and web page content by automatically crawling the provided URL.

## Features

- 🚀 **Multilingual Support**: Works with multiple languages using NVIDIA's pre-trained model
- 🌐 **Web Content Extraction**: Built-in web crawler to extract and process content from URLs
- 🎯 **Accurate Classification**: Classifies text into 25+ domain categories
- ⚡ **Fast Inference**: Optimized for quick predictions
- 🔌 **Easy Integration**: Simple Python API for seamless integration into your projects

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/oasm-platform/oasm-domain-classifier.git
   cd oasm-domain-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

## Usage

### Basic Usage

```python
from domain_classifier import DomainClassifier

# Initialize the classifier
classifier = DomainClassifier()

# Classify text
result = classifier.predict("Your text here")
print(f"Predicted domain: {result['predicted_domain']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Classify Web Content

```python
from domain_classifier import DomainClassifier
from crawl_web import CrawlWeb

# Initialize components
classifier = DomainClassifier()
crawler = CrawlWeb()

# Fetch and classify web content
text = crawler.crawl("https://example.com")
if text:
    result = classifier.predict(text)
    print(f"Predicted domain: {result['predicted_domain']}")
```

### Command Line Interface

```bash
python main.py
```

## Available Domains

The classifier can predict the following domains (non-exhaustive list):

```
'Adult', 'Arts_and_Entertainment', 'Autos_and_Vehicles', 'Beauty_and_Fitness', 'Books_and_Literature', 'Business_and_Industrial', 'Computers_and_Electronics', 'Finance', 'Food_and_Drink', 'Games', 'Health', 'Hobbies_and_Leisure', 'Home_and_Garden', 'Internet_and_Telecom', 'Jobs_and_Education', 'Law_and_Government', 'News', 'Online_Communities', 'People_and_Society', 'Pets_and_Animals', 'Real_Estate', 'Science', 'Sensitive_Subjects', 'Shopping', 'Sports', 'Travel_and_Transportation'
```

## Model Details

This package uses NVIDIA's `nvidia/multilingual-domain-classifier` model, which is based on a transformer architecture fine-tuned for domain classification across multiple languages.

## Performance

- **Inference Speed**: ~100ms per prediction (on CPU, may vary with text length)
- **Model Size**: ~1.5GB (downloaded on first use)
- **Supported Languages**: Multiple (optimized for widely-used languages)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [NVIDIA](https://huggingface.co/nvidia) for the pre-trained multilingual domain classifier model
- [Hugging Face](https://huggingface.co/) for the Transformers library
