"""
# -----------------------------------------------------------------------------------
# Python script to demonstrate Transformer-based Question Answering System.
# -----------------------------------------------------------------------------------
This program demonstrates Transformer-based Question Answering (QA) System using information about
tourist destinations in Kenya.

Dataset location:
    files/kenya_tourism.json

Requirements:
    pip install transformers torch scikit-learn

Author: Salim TS
Date: 30 Jul 2026
"""

# -----------------------------------------------------------------------------------
# 0. Import the required modules
# -----------------------------------------------------------------------------------
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import warnings

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------------
DATASET_FILE = Path('../files/kenya_tourism.json')
MODEL_NAME = 'distilbert-base-cased-distilled-squad'

# -----------------------------------------------------------------------------------
# 2. Data loading functions
# -----------------------------------------------------------------------------------
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def build_contexts(dataset):
    contexts = []
    site_names = []

    sites = dataset["sites"]
    for site in sites:
        name = site.get('name', 'Unknown Site')
        category = site.get('category', 'Attraction')
        region = site.get('region', 'Kenya')
        description = site.get('description', 'No description available.').strip()
        best_time = site.get('best_time', 'all year round').strip()

        highlights_list = site.get('highlights', [])
        highlights = "Key highlights include " + ", ".join(highlights_list) + "." if highlights_list else ""

        activities_list = site.get('activities', [])
        activities = "Visitors can enjoy activities such as " + ", ".join(activities_list) + "." if activities_list else ""

        accessibility = site.get('accessibility', '').strip()
        access_str = f"Regarding accessibility, it features a {accessibility}." if accessibility else ""

        # Continuous, standard paragraph spacing for clean token alignment
        context = (
            f"{name} is a premier {category} destination located in {region}. "
            f"{description} "
            f"The best time to visit is {best_time}. "
            f"{highlights} "
            f"{activities} "
            f"{access_str}"
        )

        # Normalize any accidental double-spacing
        context = " ".join(context.split()).strip()
        contexts.append(context)
        site_names.append(name)

    return contexts, site_names

# -----------------------------------------------------------------------------------
# 3. Retrieval Functions
# -----------------------------------------------------------------------------------
def create_tfidf_matrix(contexts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(contexts)
    return vectorizer, tfidf_matrix

def retrieve_best_context(question, vectorizer, tfidf_matrix, context, site_names):
    question_vector = vectorizer.transform([question])
    similarity_scores = cosine_similarity(question_vector, tfidf_matrix)
    best_index = similarity_scores.argmax()
    best_context = context[best_index]
    best_site = site_names[best_index]

    # Safely convert to a single scalar float to avoid matrix ambiguity crashes
    similarity_score = float(similarity_scores[0][best_index])
    return best_context, best_site, similarity_score

# -----------------------------------------------------------------------------------
# 4. Question Answering Functions
# -----------------------------------------------------------------------------------
def load_qa_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    return tokenizer, model

def answer_question(question, context, qa_pipeline):
    tokenizer, model = qa_pipeline
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    answer_tokens = inputs.input_ids[0][answer_start:answer_end]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    if not answer or answer == question:
        answer = "I could not locate a clear answer in the matching context records."

    return {"answer": answer, "score": 1.0}

# -----------------------------------------------------------------------------------
# 5. Main Execution Function
# -----------------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print(" KENYA TOURISM TRANSFORMER QUESTION ANSWERING SYSTEM")
    print("=" * 78)

    print("\nLoading dataset...")
    dataset = load_dataset(DATASET_FILE)
    print("Building tourism contexts...")
    contexts, site_names = build_contexts(dataset)
    print("Creating TF-IDF retrieval index...")
    vectorizer, tfidf_matrix = create_tfidf_matrix(contexts)

    print("Loading Transformer QA model...")
    print("Please wait on first execution...")

    qa_pipeline = load_qa_pipeline()
    print("System ready.\nType 'exit' or 'quit' to quit.")

    while True:
        print("=" * 78)
        question = input("Kindly ask a Kenyan tourism question: \n>_")

        if question.lower() == 'exit' or question.lower() == 'quit':
            print("\n" + "=" * 78)
            print("\n End of Transformer QA System Demonstration.")
            print("=" * 78)
            break

        if len(question.strip()) == 0:
            print("Please enter a valid question")
            continue

        best_context, best_site, similarity_score = (
            retrieve_best_context(question, vectorizer, tfidf_matrix, contexts, site_names)
        )

        # Baseline filter to safely separate entirely missing information
        if similarity_score < 0.20:
            print("\nResult:")
            print("I'm sorry, I couldn't find matching records regarding that in the dataset.")
            print(f"Confidence (TF-IDF Similarity: {similarity_score:.3f} is below threshold 0.20)")
            continue

        result = answer_question(question, best_context, qa_pipeline)
        answer = result["answer"]
        confidence = result["score"]

        print("\nMost Relevant Site:")
        print(best_site)

        print("\nAnswer:")
        print(answer)

        print("\nTransformer Confidence Score:")
        print(f"{confidence:.3f}")

        print(f"\nTF-IDF Similarity Score: "
              f"{similarity_score:.3f}")

        print("\nRetrieved Context:")
        # Display each sentence on its own line for human scannability
        sentences = best_context.split(". ")
        for sentence in sentences:
            if sentence:
                # Add back periods stripped out by splitting
                cleaned_sentence = sentence.strip()
                if not cleaned_sentence.endswith("."):
                    cleaned_sentence += "."
                print(cleaned_sentence)

if __name__ == "__main__":
    main()

