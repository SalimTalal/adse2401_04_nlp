"""
--------------------------------------------------------------------------------------------------
Python script to demonstrate Transfer Learning for Text classification using SpaCy & Transformers
--------------------------------------------------------------------------------------------------

This example demonstrates TRANSFER LEARNING in NLP using:
  - spaCy
  - Hugging Face Transformers
  - spaCy-Transformers

Instead of training a language model from scratch, we:
  1. Load a PRE-TRAINED transformer model (DistilBERT)
  2. Attach a text classification head
  3. Fine-tune it on a small custom dataset

Task:
  Binary sentiment classification:
    - POSITIVE
    - NEGATIVE

Task Pipeline:
1. Dependency checking
2. Building a transformer-based spaCy pipeline
3. Preparing training examples
4. Fine-tuning the classifier
5. Running inference on unseen text
6. Saving and loading the trained model

 Requirements:
    - pip install -U transformers torch spacy-transformers

Author: Salim TS
Date: 22 July 2026
"""

#----------------------------------------------------------
# 0. Import the required modules
#----------------------------------------------------------
from __future__ import annotations # Ensure this is the 1st import to avoid getting errors
import random, sys
from pathlib import Path
from typing import Any

#----------------------------------------------------------
# 1. Dependency checks
#----------------------------------------------------------
def check_import(module_name: str, install_hint: str) -> Any:
    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"\n[ERROR]: Missing dependency: {module_name}"
              f"\nInstall using:\n {install_hint}\n")
        sys.exit(1)

# Core Libraries
spacy = check_import('spacy','pip install spacy')
transformer = check_import('transformers','pip install transformers')

#from spacy.training.example import Example

#----------------------------------------------------------
# 2. Training data
#----------------------------------------------------------
TRAIN_DATA = [
    (
        "I absolutely love this product",
        {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}},
    ),
    (
        "This movie was fantastic",
        {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}},
    ),
    (
        "The service was excellent",
        {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}},
    ),
    (
        "I hate this item",
        {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}},
    ),
    (
        "This was a terrible experience",
        {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}},
    ),
    (
        "The food tasted awful",
        {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}},
    ),
]

#----------------------------------------------------------
# 3. Build spaCy pipeline with TRANSFORMER
#----------------------------------------------------------
print("\n[INFO]: Creating NLP Pipeline")

# Create blank English pipeline
nlp = spacy.blank("en")

# Add transformer component(Transfer learning happens here using a pretrained Hugging Face model).
nlp.add_pipe(
    "transformer",
    config={
        "model":{
            "@architectures":"spacy-transformers.TransformerModel.v3",
            "name":"distilbert-base-uncased"
        }
    }
)

# Add text classifier (Model spaCy versions autoconfigure the architectures)
textcat = nlp.add_pipe("textcat",last=True)

# Add labels
textcat.add_label("POSITIVE")
textcat.add_label("NEGATIVE")
print("\n[INFO]: Pipeline components"
      f"    {nlp.pipe_names}\n")

#----------------------------------------------------------
# 4. Initialise Training
#----------------------------------------------------------

#----------------------------------------------------------
# 5.
#----------------------------------------------------------

#----------------------------------------------------------
# 6. Save Model
#----------------------------------------------------------
output_dir = Path("../files/sentiment_transformer_model")

# Create directory if missing
output_dir.mkdir(parents=True, exist_ok=True)

nlp.to_disk(output_dir)

# Display save location
print(f"\n[INFO]: Saved model to : \n{output_dir}")

#----------------------------------------------------------
# 7. Inference / Prediction
#----------------------------------------------------------
print("INFERENCE DEMO...\n")

TEST_TEXTS = [
    "I really enjoyed this book",
    "The customer support was horrible",
    "Amazing performance by the actors",
    "This app is frustrating and buggy"
]

for text in TEST_TEXTS:

    doc = nlp(text)

    positive_core = doc.cats["POSITIVE"]
    negative_core = doc.cats["NEGATIVE"]

    predicted = max(doc.cats, key=doc.cats.get)

    print("\n" + "-" * 55)
    print(f"TEXT        : {text}")
    print(f"PREDICTION  : {predicted}")
    print(f"POSITIVE    : {positive_core:.3f}")
    print(f"NEGATIVE    : {negative_core:.3f}")

#----------------------------------------------------------
# 8. Optional: Reload saved model
#----------------------------------------------------------
print(f"\n" + "-" * 55)
print("MODEL RELOAD DEMO")
print(f"=" * 55)
loaded_nlp = spacy.load(output_dir)
reload_doc = loaded_nlp("This laptop is amazing!")

print("\nReloaded model prediction: {reload_doc.cats}")
