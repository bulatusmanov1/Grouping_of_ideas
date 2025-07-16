import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils.jsonl import *
from utils.embedding import *
from utils.transform import *

def compute_embeddings(texts: List[str], model_name: str = 'cointegrated/LaBSE-en-ru') -> np.ndarray:
    """
    Эмбеддинги для списка из строк в full_text
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def cluster_embeddings(idea_ids, embeddings, eps=0.25, min_samples=2):
        clustering = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(embeddings)

        df = pd.DataFrame({
            'idea_id': idea_ids,
            'cluster_id': labels
        })
        return df

def match_new_idea_to_old_db(
    new_text: str,
    db,
    top_n: int = 15,
    model_name: str = 'cointegrated/LaBSE-en-ru'
) -> Tuple[List[Tuple[str, str, float]], Dict]:
    """
    Принимает новую идею и:
    - Возвращает топ-N похожих идей [(idea_id, полный_текст, %_сходства), ...]
    - Самый похожий кластер из таблицы clusters
    """
    new_key_words = get_key_words([new_text])
    new_key_words_filtered = filter_organizations_spacy(new_key_words[0])
    new_cleaned_text = get_clean_text([new_text], [new_key_words_filtered])[0]

    model = SentenceTransformer(model_name)
    new_embedding = model.encode([new_cleaned_text], convert_to_numpy=True)[0]

    db.cursor.execute('SELECT idea_id, idea_title, idea_description, idea_embedding FROM ideas')
    rows = db.cursor.fetchall()

    idea_ids = []
    full_texts = []
    old_embeddings = []

    idea_id_to_embedding = {}

    for idea_id, title, description, embedding in rows:
        if embedding is None:
            continue
        idea_ids.append(idea_id)
        full_texts.append(f"{title.strip()} {description.strip()}")
        old_embeddings.append(embedding)
        idea_id_to_embedding[str(idea_id)] = embedding

    if not old_embeddings:
        return [], {}

    old_embeddings = np.array(old_embeddings)
    similarities = cosine_similarity([new_embedding], old_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in ranked_indices[:top_n]:
        matched_text = full_texts[idx]
        idea_id = idea_ids[idx]
        similarity_percent = round(similarities[idx] * 100, 2)
        results.append((idea_id, matched_text, similarity_percent))

    db.cursor.execute("SELECT cluster_id, clusters FROM clusters")
    cluster_rows = db.cursor.fetchall()

    best_cluster = None
    best_score = -1

    for cluster_id, cluster_idea_ids in cluster_rows:
        cluster_vectors = [
            idea_id_to_embedding[str(idea_id)]
            for idea_id in cluster_idea_ids
            if str(idea_id) in idea_id_to_embedding
        ]

        if not cluster_vectors:
            continue

        cluster_mean_vector = np.mean(cluster_vectors, axis=0)
        score = cosine_similarity([new_embedding], [cluster_mean_vector])[0][0]

        if score > best_score:
            best_score = score
            best_cluster = {
                "cluster_id": cluster_id,
                "idea_ids": cluster_idea_ids,
                "similarity": round(score * 100, 2)
            }

    return results, best_cluster if best_cluster else {}

def match_new_idea_to_old_jsonl(
    new_text: str,
    df: pd.DataFrame,
    top_n: int = 15,
    model_name: str = 'cointegrated/LaBSE-en-ru',
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