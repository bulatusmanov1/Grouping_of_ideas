import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


def compute_embeddings(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Эмбеддинги для списка из строк в full_text
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def cluster_embeddings(
    idea_ids: List[str],
    embeddings: np.ndarray,
    eps: float = 0.25,
    min_samples: int = 2
) -> pd.DataFrame:
    """
    Кластеризует эмбеддинги с помощью DBSCAN и возвращает DataFrame с метками кластеров.
    """
    print("Выполняется кластеризация эмбеддингов (DBSCAN)...")
    clustering = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(embeddings)

    df = pd.DataFrame({
        'idea_id': idea_ids,
        'cluster_id': labels
    })

    print("Кластеризация завершена")
    return df