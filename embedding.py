import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from utils import *
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
    #print("Выполняется кластеризация эмбеддингов (DBSCAN)...")
    clustering = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(embeddings)

    df = pd.DataFrame({
        'idea_id': idea_ids,
        'cluster_id': labels
    })

    #print("Кластеризация завершена")
    return df



def find_best_subgroup_for_new_idea(new_text: str, cluster_subgroups_contexts: Dict[int, List[List[List[str]]]], threshold=20.0):
    """
    Находит подгруппу, куда попадёт новая идея, при добавлении её в каждую существующую подгруппу всех кластеров.
    """
    print("[1] Обработка новой идеи...")
    print("[debug] Вызвана find_best_subgroup_for_new_idea")
    
    context = get_key_words([new_text])[0]
    filtered_context = filter_organizations_spacy(context)
    clean_text = get_clean_text([new_text], [filtered_context])[0]
    tokens_new = filtered_context if filtered_context else ['АРГЕС']

    for cluster_id, subgroups in cluster_subgroups_contexts.items():
        for i, subgroup_contexts in enumerate(subgroups):
            contexts_with_new = subgroup_contexts + [tokens_new]
            subgroups_new = smart_grouping(contexts_with_new, threshold=threshold)

            for subgroup in subgroups_new:
                if len(subgroup) > 1 and (len(contexts_with_new) - 1) in subgroup:
                    matched_contexts = [subgroup_contexts[idx] for idx in subgroup if idx != len(contexts_with_new) - 1]
                    
                    print(f"\n✅ Идея попала в кластер {cluster_id}, подгруппу {i + 1}")
                    print("Похожие идеи в подгруппе:")
                    for ctx in matched_contexts:
                        print("—", ";".join(ctx))
                    
                    return cluster_id, i + 1, matched_contexts

    print("❌ Идея не попала ни в одну подгруппу")
    return None




