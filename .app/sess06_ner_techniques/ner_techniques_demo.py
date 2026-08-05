"""
-----------------------------------------------------------------------------------
Python script to various Named Entity Recognition (NER) techniques.
-----------------------------------------------------------------------------------
This script demonstrates the following NER techniques
1. Rule-Based NER
2. CRF-based NER
3. spaCy Pretrained Transformer/Statistical NER
4. HuggingFace Transformer-based NER

Each  technique is evaluated using
- Precision
- Recall
- F1-Score
- Confusion Matrix

-----------------------------------------------------------------------------------
Required Modules
-----------------------------------------------------------------------------------
pip install spacy sklearn-crfsuite transformers torch seqeval datasets

Download the spaCy medium model
python -m spacy download en_core_web_md

-----------------------------------------------------------------------------------
DATASET FORMAT
-----------------------------------------------------------------------------------

The dataset uses BIO tagging format.

Example:
[
    ("Chuck", "B-PER"),
    ("Missler", "I-PER"),
    ("visited", "O"),
    ("Kenya", "B-LOC")
]

-----------------------------------------------------------------------------------
NOTES
-----------------------------------------------------------------------------------

This script is educational and intentionally simplified.

- Rule-based NER is heuristic only.
- CRF uses handcrafted features.
- spaCy uses pretrained statistical/deep learning.
- Transformers use HuggingFace pipelines.
-----------------------------------------------------------------------------------

Author: Salim TS
Date: 26 July 2026
"""
#-----------------------------------------------------------------------------------
# 0. Import the required modules
#-----------------------------------------------------------------------------------
import re, spacy, sys, subprocess
from collections import defaultdict
from seqeval.metrics import(
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from datasets import load_dataset
from spacy.util import is_package
from transformers import pipeline

#-----------------------------------------------------------------------------------
# 1. Sample Dataset
#-----------------------------------------------------------------------------------
train_data = [
    [
        ("Chuck", "B-PER"),
        ("Missler", "I-PER"),
        ("visited", "O"),
        ("Kenya", "B-LOC")
    ],
    [
        ("Microsoft", "B-ORG"),
        ("is", "O"),
        ("based", "O"),
        ("in", "O"),
        ("Seattle", "B-LOC")
    ],
    [
        ("Elon", "B-PER"),
        ("Musk", "I-PER"),
        ("founded", "O"),
        ("SpaceX", "B-ORG")
    ]
]

test_data = [
    [
        ("Jeff", "B-PER"),
        ("Bezos", "I-PER"),
        ("owns", "O"),
        ("Amazon", "B-ORG")
    ],
    [
        ("Google", "B-ORG"),
        ("opened", "O"),
        ("an", "O"),
        ("office", "O"),
        ("in", "O"),
        ("Nairobi", "B-LOC")
    ]
]

#-----------------------------------------------------------------------------------
# 2. Utility Functions
#-----------------------------------------------------------------------------------
def extract_tokens(dataset):
  return [[token for token, label in sentence] for sentence in dataset]

def extract_labels(dataset):
  return [[label for token, label in sentence] for sentence in dataset]


#-----------------------------------------------------------------------------------
# 3. Evaluate Function
#-----------------------------------------------------------------------------------
def evaluate_model(true_labels, predicted_labels, model_name):
  print("\n" + "=" * 60)
  print(f"EVALUATION {model_name}")
  print("-" * 60)

  precision = precision_score(true_labels, predicted_labels)
  recall = recall_score(true_labels, predicted_labels)
  f1 = f1_score(true_labels, predicted_labels)

  print(f"Precision: {precision:.3f}")
  print(f"Recall: {recall:.3f}")
  print(f"F1-Score: {f1:.3f}")

  print(f"\nDetailed Report:")
  print(classification_report(true_labels, predicted_labels))

  print("\n" + "=" * 60)


#-----------------------------------------------------------------------------------
# 4. (i) Rule_Based NER class
#-----------------------------------------------------------------------------------
class RuleBaseNER:
  def __init__(self):
    self.person_titles = {"Mr","Mrs","Dr"}

    self.locations = {
        "Kenya",
        "Seattle",
        "Nairobi"
    }

    self.organisations = {
        "Microsoft",
        "Google",
        "Amazon",
        "SpaceX"
    }

  def predict(self, sentence_tokens):

    predictions = []
    for token in sentence_tokens:
        if token in self.locations:
            predictions.append("B-LOC")

        elif token in self.organisations:
            predictions.append("B-ORG")

        elif token[0].isupper():
            predictions.append("B-PER")

        else:
            predictions.append("O") # Changed from "0" to "O"
    return predictions

#-----------------------------------------------------------------------------------
# 4. (ii) CFR-Based NER function
#-----------------------------------------------------------------------------------
def word2features(sentence,index):

    word = sentence[index][0]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit()
    }

    # Previous word
    if index > 0:
        previous_word = sentence[index-1][0]

        features.update({
            "-1:word.lower()": previous_word.lower(),
            "-1:word.istitle()": previous_word.istitle()
        })
    else:
        features["BOS"] = True

    # Next Word
    if index < len(sentence)-1:
        next_word = sentence[index+1][0]

        features.update({
            "1:word.lower()": next_word.lower(),
            "1:word.istitle()": next_word.istitle()
        })
    else:
        features["EOS"] = True

    return features

def sent2features(sentence):

    return [word2features(sentence,n)for n in range(len(sentence))]

def sent2labels(sentence):

    return [label for token, label in sentence]

#-----------------------------------------------------------------------------------
# 4. (iii) SpaCy NER class
#-----------------------------------------------------------------------------------
class SpacyNER:

    def __init__(self):

        self.model_name = 'en_core_web_md'

        # Check whether the model exists
        if not is_package(self.model_name):

            print(f"{self.model_name} not available!"
                  f"\nDownloading spaCy medium model...")

            subprocess.check_call([
                sys.executable,
                "-m",
                "spacy",
                "download",
                self.model_name
            ])

            # notify of a successfull download
            print(f"{self.model_name} downloaded successfully!")

        # Load the model
        self.nlp = spacy.load(self.model_name)

    def predict(self,tokens):

        text = " ".join(tokens)

        doc = self.nlp(text)

        predictions = ["O"] * len(doc) # Changed from "0" to "O"

        for ent in doc.ents:
            for idx, token in enumerate(ent):
                label = ent.label_

                mapped_label = {
                    "PERSON" : "PER",
                    "ORG" : "ORG",
                    "GPE" : "GPE"
                }.get(label,None)

                if mapped_label:
                    token_index = token.i

                    if idx == 0:
                        predictions[token_index] = f"B-{mapped_label}"
                    else:
                        predictions[token_index] = f"I-{mapped_label}"

        return predictions
#-----------------------------------------------------------------------------------
# 4. (iv) Transformer-Based NER class
#-----------------------------------------------------------------------------------
class TransformerNER:

    def __init__(self):

        self.ner_pipeline = pipeline(
            "ner",
            aggregation_strategy="simple",
            model="dslim/bert-base-NER"
        )

    def predict(self,tokens):

        text = " ".join(tokens)

        entities = self.ner_pipeline(text)

        # Initialize predictions with the correct length (number of input tokens)
        predictions = ["O"] * len(tokens)

        # Create a mapping from original tokens to their character start/end positions
        # to correctly align pipeline entities with original tokens.
        token_char_spans = []
        current_char_idx = 0
        for token in tokens:
            token_char_spans.append((current_char_idx, current_char_idx + len(token)))
            current_char_idx += len(token) + 1 # +1 for the space

        for entity in entities:
            entity_start = entity["start"]
            entity_end = entity["end"]
            entity_label = entity["entity_group"]

            # Map entity label to standard format if necessary
            tag = None
            if entity_label == "PER":
                tag = "PER"
            elif entity_label == "ORG":
                tag = "ORG"
            elif entity_label == "LOC":
                tag = "LOC"
            else:
                continue # Skip entities with unmapped labels

            # Find the tokens corresponding to the current entity
            entity_token_indices = []
            for i, (token_start, token_end) in enumerate(token_char_spans):
                # Check for overlap between entity span and token span
                if max(entity_start, token_start) < min(entity_end, token_end):
                    entity_token_indices.append(i)

            if entity_token_indices:
                # Assign B- tag to the first token of the entity
                predictions[entity_token_indices[0]] = f"B-{tag}"
                # Assign I- tag to subsequent tokens of the entity
                for i in entity_token_indices[1:]:
                    predictions[i] = f"I-{tag}"

        return predictions

#-----------------------------------------------------------------------------------
# 5. Main Execution Function
#-----------------------------------------------------------------------------------
def main() -> None:
    """
    Train and evaluate all NER techniques/approaches.

    Workflow:
    1. Prepare CRF features
    2. Train CRF model
    3. Generate predictions from
        i) Rule-based model
        ii) CRF model
        iii) spaCy model
        iv) Transformer-based model
    """
    X_train = [sent2features(s) for s in train_data]
    y_train = [sent2labels(s) for s in train_data]

    X_test = [sent2features(s) for s in test_data]
    y_test = [sent2labels(s) for s in test_data]

    token_test = extract_tokens(test_data)

    # -----------------------------------------------------------------------------------
    # I. Rule-Based NER
    # -----------------------------------------------------------------------------------
    rule_ner = RuleBaseNER()

    rule_predictions = [
        rule_ner.predict(tokens)
        for tokens in token_test
    ]

    evaluate_model(
        y_test,
        rule_predictions,
        "Rule-based NER"
    )
    # -----------------------------------------------------------------------------------
    # II. CRF-Based NER
    # -----------------------------------------------------------------------------------

    crf = CRF(
        algorithm ="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100
    )

    crf.fit(X_train, y_train)

    crf_predictions = crf.predict(X_test)

    evaluate_model(
        y_test,
        crf_predictions,
        "CRF-Based NER"
    )
    # -----------------------------------------------------------------------------------
    # III. SPACY NER
    # -----------------------------------------------------------------------------------
    spacy_ner = SpacyNER()

    spacy_predictions = [
        spacy_ner.predict(tokens)
        for tokens in token_test
    ]

    evaluate_model(
        y_test,
        spacy_predictions,
        "spaCy NER"
    )
    # -----------------------------------------------------------------------------------
    # IV. Transformer-Based NER
    # -----------------------------------------------------------------------------------

    transformer_ner = TransformerNER()

    transformer_predictions = [
        transformer_ner.predict(tokens)
        for tokens in token_test
    ]

    evaluate_model(
        y_test,
        transformer_predictions,
        "Transformer-based NER"
    )

    print("\n[INFO]: Done. NER techniques Demo Complete.")

# -----------------------------------------------------------------------------------
# 6. Run the script by invoking it's main() function
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    main()