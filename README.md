# Domain Classifier

A Python-based tool for classifying website domains by analyzing their content using pre-trained transformer models from Hugging Face.

## Features

- Classify website domains based on their content
- Simple command-line interface
- Supports direct URL input
- Efficient text extraction and preprocessing
- Optimized for both CPU and GPU

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

Classify a website domain:

```bash
python main.py
```

This will classify the default URL (24h.com.vn). To classify a different website, modify the `url` variable in `main.py`.

### Example Output

```
Loading model components...
Model loaded successfully! Using device: CPU

Fetching content from: 24h.com.vn
Classifying content...

Predicted domain category: Business_and_Industrial
```

## Project Structure

```
oasm-domain-classifier/
├── classifier.py    # Domain classifier implementation
├── model.py         # Custom model implementation
├── main.py          # Main script
├── setup.py         # Setup script
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- BeautifulSoup4
- Requests

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
