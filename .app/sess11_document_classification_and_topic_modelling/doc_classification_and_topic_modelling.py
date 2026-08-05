"""
===================================================================================================================
Python script to demonstrate a Document Classification and Topic Modeling.
===================================================================================================================
This program demonstrates two natural language process tasks i.e. Document Classification (Supervised Learning)
and Topic Modeling (Unsupervised Learning).

PART 1: Document Classification (Supervised Learning)
- TF-IDF Vectorization
- Train/Test Split
- Multinomial Naive Bayes Classification
- Accuracy Evaluation
- Classification Report
- Interaction Predictions

PART 2: Topic Modeling (Unsupervised Learning)
- TF-IDF Vectorization
- Latent Dirichlet Allocation (LDA)
- Topic Discovery
- Topic Interpretation
- Dominant Topic Assignment

Dataset location:
    files/articles.json
    files/topics.json

Requirements:
    !pip install scikit-learn pandas numpy

Author: Salim TS
Date: 02 Aug 2026
"""
# -----------------------------------------------------------------------------------
# 0. Import the required modules
# -----------------------------------------------------------------------------------
import os, json, re, sys, warnings, numpy as np, pandas as pd

from pathlib import Path
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------
# 1.
# -----------------------------------------------------------------------------------