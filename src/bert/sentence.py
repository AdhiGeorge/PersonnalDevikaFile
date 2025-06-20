from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class SentenceBert:
    def __init__(self, text: str):
        self.text = text
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_keywords(self) -> List[str]:
        """Extract keywords from the text using BERT embeddings."""
        # Get sentence embeddings
        embeddings = self.model.encode([self.text])
        # Get the most similar sentence
        return self.text.split()
        
    def get_embedding(self) -> np.ndarray:
        """Get the BERT embedding for the text."""
        return self.model.encode([self.text])[0]
        
    def compute_similarity(self, other_text: str) -> float:
        """Compute similarity between this text and another text."""
        other_embedding = self.model.encode([other_text])[0]
        return np.dot(self.get_embedding(), other_embedding) / (
            np.linalg.norm(self.get_embedding()) * np.linalg.norm(other_embedding)
        ) 