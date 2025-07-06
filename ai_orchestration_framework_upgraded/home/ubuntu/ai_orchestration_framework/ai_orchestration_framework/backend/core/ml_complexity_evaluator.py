
import re
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

from .model_connectors import TaskComplexity
from .base_evaluator import ComplexityEvaluator # Changed to import from base_evaluator.py

class MLComplexityEvaluator(ComplexityEvaluator):
    def __init__(self, model_path="complexity_model.joblib", vectorizer_path="tfidf_vectorizer.joblib"):
        self.model = None
        self.vectorizer = None
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            print("Loaded existing ML complexity model and vectorizer.")
        else:
            print("No existing ML complexity model found. Model will be trained on first use.")

    def _get_wordnet_pos(self, tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN # Default to noun

    def _extract_features(self, prompt: str) -> dict:
        tokens = word_tokenize(prompt.lower())
        sentences = sent_tokenize(prompt)
        pos_tags = pos_tag(tokens)

        # Syntactic Features
        num_sentences = len(sentences)
        avg_sentence_length = np.mean([len(word_tokenize(s)) for s in sentences]) if num_sentences > 0 else 0
        num_words = len(tokens)

        # POS Tag Distribution
        pos_counts = {"NN": 0, "VB": 0, "JJ": 0, "RB": 0}
        for word, tag in pos_tags:
            if tag.startswith("NN"):
                pos_counts["NN"] += 1
            elif tag.startswith("VB"):
                pos_counts["VB"] += 1
            elif tag.startswith("JJ"):
                pos_counts["JJ"] += 1
            elif tag.startswith("RB"):
                pos_counts["RB"] += 1
        
        # Readability (simplified Flesch-Kincaid for demonstration)
        # This is a very basic approximation. Real readability scores are more complex.
        avg_syllables_per_word = np.mean([len(re.findall(r"[aeiouy]+", word.lower())) for word in tokens]) if num_words > 0 else 0
        readability_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word

        return {
            "num_sentences": num_sentences,
            "avg_sentence_length": avg_sentence_length,
            "num_words": num_words,
            "pos_nn_ratio": pos_counts["NN"] / num_words if num_words > 0 else 0,
            "pos_vb_ratio": pos_counts["VB"] / num_words if num_words > 0 else 0,
            "pos_jj_ratio": pos_counts["JJ"] / num_words if num_words > 0 else 0,
            "pos_rb_ratio": pos_counts["RB"] / num_words if num_words > 0 else 0,
            "readability_score": readability_score,
        }

    async def evaluate(self, prompt: str, task_type: str = "general") -> TaskComplexity:
        if self.model is None or self.vectorizer is None:
            # If model not loaded, train a dummy one for demonstration
            print("Training dummy ML complexity model...")
            # Dummy training data (replace with real data for production)
            prompts = [
                "Summarize this document briefly.", # SIMPLE
                "What are the main points of the article?", # SIMPLE
                "Compare and contrast the two economic theories.", # MEDIUM
                "Analyze the impact of climate change on global economies.", # MEDIUM
                "Design a scalable microservices architecture for a high-traffic e-commerce platform, considering security and performance.", # COMPLEX
                "Implement a machine learning algorithm for real-time anomaly detection in network traffic data.", # COMPLEX
            ]
            complexities = [
                TaskComplexity.SIMPLE, TaskComplexity.SIMPLE,
                TaskComplexity.MEDIUM, TaskComplexity.MEDIUM,
                TaskComplexity.COMPLEX, TaskComplexity.COMPLEX
            ]

            # Extract features for training
            X_features = [self._extract_features(p) for p in prompts]
            # Convert dicts to a feature matrix (simple approach for demo)
            # In a real scenario, you'd use DictVectorizer or similar
            feature_names = list(X_features[0].keys())
            X = np.array([[f[name] for name in feature_names] for f in X_features])
            y = np.array([c.value for c in complexities])

            # TF-IDF for text content
            self.vectorizer = TfidfVectorizer()
            X_tfidf = self.vectorizer.fit_transform(prompts)

            # Combine features (simple concatenation for demo)
            # In a real scenario, you'd need a more robust feature union
            X_combined = np.hstack((X, X_tfidf.toarray()))

            self.model = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))
            self.model.fit(X_combined, y)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.vectorizer, self.vectorizer_path)
            print("Dummy ML complexity model trained and saved.")

        # Predict complexity for the given prompt
        features = self._extract_features(prompt)
        feature_names = list(features.keys())
        X_new = np.array([[features[name] for name in feature_names]])
        X_new_tfidf = self.vectorizer.transform([prompt])
        X_new_combined = np.hstack((X_new, X_new_tfidf.toarray()))

        prediction_value = self.model.predict(X_new_combined)[0]
        return TaskComplexity(prediction_value)


# Example Usage (for testing)
async def main():
    evaluator = MLComplexityEvaluator()
    print(await evaluator.evaluate("Summarize this document."))
    print(await evaluator.evaluate("Analyze the market trends for renewable energy."))
    print(await evaluator.evaluate("Develop a secure and scalable blockchain solution for supply chain management."))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


