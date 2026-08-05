# Python script to demonstrate
# Natural Language Preprocessing pipeline for Restaurant Reviews
"""
The preprocessing pipeline includes:
1. Lowercasing
2. Slang and abbreviation normalization
3. Contraction expansion
4. Repeated character normalization
5. Emoji removal
6. Punctuation cleaning
7. Tokenization
8. Stopword Removal
9. Optimal Lemmatization

Author: Salim TS
Date: 20 Jul 2026
"""
#----------------------------------------------------------
# 0. Import required modules
#----------------------------------------------------------
import matplotlib.pyplot as plt
import nltk
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List

#----------------------------------------------------------
# 1. Download the required data
#----------------------------------------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

#----------------------------------------------------------
# 2. Raw Data ( restaurant reviews )
#----------------------------------------------------------
REVIEWS = [
    "The food was 2die4! Best burgers in town 😋🍔",
    "Service was 10/10, loved the vibe too! Totally recommend this place 🔥👌",
    "2 hrs wait for a table... not worth it. Food was ok, but not great.",
    "Tbh, 4 the price, I expected way better quality. Disappointed.",
    "Great place for brunch! Had the eggs benedict, 1 of my faves 💯🍳",
    "The ambience was gr8! Luvd it :) !!!",
    "Had a blast at this place!! Will come again soon 😋",
    "Food was OK, but could be better, meh...",
    "This pizza was absolutely amazing, best I’ve had!!",
    "Service was horrible... never coming back!",
    "I loved the pasta! But the portion was so small :(",
    "The dessert was soooooo good!! 😍",
    "So disappointed, the steak was overcooked...",
    "Great experience, but the music was a little loud tbh.",
    "Good food, but they forgot my drink. :(",
    "Superb food! Totally worth the price! Will return!",
    "Was okay, nothing special. Meh.",
    "The chicken was so dry, I couldn't finish it :( too bad!",
    "Fantastic, I can't wait to visit again! :) :)",
    "Not worth the price, won't be coming back :( 😔"
]

#----------------------------------------------------------
# 3. Normalisation Rules
#----------------------------------------------------------
SLANG_DICT = {
    r'\b2die4\b': 'to die for',
        r'\bgr8\b': 'great',
        r'\bluvd\b': 'loved',
        r'\bfaves\b': 'favourites',
        r'\btbh\b': 'to be honest',
        r'\bthx\b': 'thanks',
        r'\bplz\b': 'please',
        r'\bu\b': 'you',
        r'\b4\b': 'for',
        r'\b2\b': 'to',
        r'\b1\b': 'one',
        r'\b10/10\b': 'perfect',
        r'\bok\b': 'okay',
        r'\bmeh\b': 'mediocre',
        r'\btotally\s*recommend\b': 'highly recommend',
        r'\bambiance\b': 'ambience',  # US to UK spelling
}

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
}

#----------------------------------------------------------
# 4. Preprocessing Functions
#----------------------------------------------------------
def normalise_text(text: str) -> str:
    """
    Apply basic text normalization.

    This includes lowercasing, slang replacement, and contraction expansion

    :param text: Raw input text
    :return: Normalised input text
    """
    text = text.lower()

    # Replace slang
    for pattern, replacement in SLANG_DICT.items():
        text = re.sub(pattern, replacement, text)

    # Expand contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)

    return text

def remove_emojis(text: str) -> str:
    """
    Remove emojis and non-ASCII text/characters from input text.

    :param text: Input text/string
    :return: Cleaned string
    """
    return re.sub(r'[^\x00-\x7F]+','', text)

def normalise_repeated_characters(text: str) -> str:
    """
    Reduce repeated characters into text (e.g. 'soooooooooo' -> 'so').

    :param text: Input text/string
    :return: Normalised string
    """
    return re.sub(r'(.)\1{2,}',r'\1\1',text)

def clean_text(text: str) -> str:
    """
    Remove punctuation and extra whitespace from input text.

    :param text: Input text/string
    :return: Cleaned string
    """
    text = re.sub(r'[^a-z\s]',' ',text)
    text = re.sub(r'\s+',' ',text)
    return text.strip()

