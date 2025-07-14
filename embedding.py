import pandas as pd
import torch
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from embedding import *
from transform import *
from sklearn.metrics.pairwise import cosine_similarity

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
    clustering = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(embeddings)

    df = pd.DataFrame({
        'idea_id': idea_ids,
        'cluster_id': labels
    })
    return df


def match_new_idea_to_old(
    new_text: str,
    df: pd.DataFrame,
    top_n: int = 15,
    model_name: str = 'all-MiniLM-L6-v2',
    grouped_path: str = 'grouped_ideas.json'
) -> Tuple[List[Tuple[str, str, float]], Dict]:
    """
    Принимает новую идею и:
    - Возвращает топ-N похожих идей [(idea_id, полный_текст, %_сходства), ...]
    - Наиболее близкую подгруппу из grouped_ideas.json
    """
    idea_ids, old_key_words, old_embeddings = json_load('embeddings.jsonl')
    old_texts = df.set_index('idea_id').loc[idea_ids]['full_text'].tolist()

    new_key_words = get_key_words([new_text])
    new_key_words_filtered = filter_organizations_spacy(new_key_words[0])
    new_cleaned_text = get_clean_text([new_text], [new_key_words_filtered])[0]

    model = SentenceTransformer(model_name)
    new_embedding = model.encode([new_cleaned_text], convert_to_numpy=True)[0]

    similarities = cosine_similarity([new_embedding], old_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]
    results = []

    for idx in ranked_indices[:top_n]:
        matched_text = old_texts[idx]
        idea_id = idea_ids[idx]
        similarity_percent = round(similarities[idx] * 100, 2)
        results.append((idea_id, matched_text, similarity_percent))

    best_group = None
    best_score = -1

    try:
        with open(grouped_path, "r", encoding="utf-8") as f:
            grouped_ideas = json.load(f)

        for group in grouped_ideas:
            group_texts = group['texts']
            group_embs = model.encode(group_texts, convert_to_numpy=True)
            group_mean_emb = np.mean(group_embs, axis=0)
            score = cosine_similarity([new_embedding], [group_mean_emb])[0][0]

            if score > best_score:
                best_score = score
                best_group = group

    except FileNotFoundError:
        best_group = {}

    return results, best_group