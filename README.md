# Cat Handwriting Recognizer

A simple Python module for recognizing handwritten using machine learning. Designed for easy integration and efficient preprocessing.

## Features

- Preprocess handwritten images
- Extract simple features for recognition
- K-Nearest Neighbors (KNN) prediction
- Async support for building templates
- Timeout mechanism for long-running operations
- Save and load models as .npz files

## Installation

Install from GitHub:

```bash
pip install git+https://github.com/littlecommandcat/cat_handwriting_recognizer.git
```

## Quick Example
```py
from cat_handwriting_recognizer import HandwritingRecognizer  
import asyncio

machine = HandwritingRecognizer(timeout=-1)

# Build templates from data
asyncio.run(machine.build_all_templates("directory")) # Your data directory

# Save model
machine.save_model("model.npz") # Save model file(.npz)

# Load model
machine.load_model("model.npz") # Load model file(.npz)

# Predict a picture
result = asyncio.run(machine.predict("file.png")) # Use model to predict png/jpg/jpeg file
print(result)
```
## Requirements

- Python 3.10+
- NumPy
- Pillow

## License

MIT License

## Author

littlecommandcat
