import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Объединяет колонки название и описание в одну + заполняет пропуски
    ['idea_id', 'full_text']
    """
    df = pd.read_csv(file_path, sep=';', dtype=str)
    df.fillna('', inplace=True)

    df['full_text'] = df['Название'].str.strip() + '. ' + df['Описание'].str.strip()
    df['full_text'] = df['full_text'].str.strip().str.replace(r'\s+', ' ', regex=True)

    df = df[df['full_text'].str.strip() != '']
    df.rename(columns={'Номер идеи': 'idea_id'}, inplace=True)
    df = df[['idea_id', 'full_text']].reset_index(drop=True)

    return df

def compute_embeddings(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Эмбеддинги для списка из строк в full_text
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def save_embeddings(df: pd.DataFrame, embeddings: np.ndarray, output_path: str) -> None:
    """
    Сохраняет эмбеддинги в CSV: idea_id + вектор эмбеддинга.
    """
    df_copy = df[['idea_id']].copy()
    df_copy['embedding'] = [emb.tolist() for emb in embeddings]
    df_copy.to_json(output_path, orient='records', lines=True, force_ascii=False)

def load_embeddings(path: str) -> Tuple[List[str], np.ndarray]:
    """
    Загружает идеи и их эмбеддинги из JSONL.
    """
    records = pd.read_json(path, lines=True)
    idea_ids = records['idea_id'].tolist()
    embeddings = np.vstack(records['embedding'].values)
    return idea_ids, embeddings

def start_p_1():
    print("Шаг 1/3: Загружаем и обрабатываем данные...")
    df = load_and_preprocess_data('data.csv')
    
    print("Шаг 2/3: Строим эмбеддинги...")
    texts = df['full_text'].tolist()
    texts = list(tqdm(texts, desc="📄 Подготовка текстов", unit="текст"))
    embeddings = compute_embeddings(texts)
    
    print("Шаг 3/3: Сохраняем эмбеддинги в файл...")
    save_embeddings(df, embeddings, 'embeddings.jsonl')
    
    print("Эмбеддинги сохранены в 'embeddings.jsonl'")

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

def extract_duplicates_and_uniques(
    df_clusters: pd.DataFrame
) -> Tuple[List[List[str]], List[str]]:
    """
    Делит идеи на кластеры повторов и уникальные.
    """
    print("Анализируем кластеры на повторы и уникальные идеи...")

    groups = df_clusters.groupby('cluster_id')['idea_id'].apply(list)

    duplicates = []
    for cid, group in tqdm(groups.items(), desc="Поиск повторов", unit="кластер"):
        if cid != -1 and len(group) > 1:
            duplicates.append(group)

    uniques = groups.get(-1, [])

    print(f"Повторяющихся групп: {len(duplicates)}")
    print(f"Уникальных идей: {len(uniques)}")

    return duplicates, uniques

def start_p_2():
    idea_ids, embeddings = load_embeddings('embeddings.jsonl')
    df_clusters = cluster_embeddings(idea_ids, embeddings, eps=0.25, min_samples=2)
    duplicate_groups, unique_ideas = extract_duplicates_and_uniques(df_clusters)
    """
    print("Найдено повторяющихся групп:", len(duplicate_groups))
    for i, group in enumerate(duplicate_groups, 1):
        print(f"Группа {i}: {group}")

    print("Уникальных идей:", len(unique_ideas))
    """
    
start_p_2()
