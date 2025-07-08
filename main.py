from utils import *
from embedding import *
from transform import *

def start_p_1():
    print("Шаг 1/3: Загружаем и обрабатываем данные...")
    df = load_and_preprocess_data('data.csv')
    
    print("Шаг 2/3: Строим эмбеддинги...")
    texts = df['full_text'].tolist()
    texts = list(tqdm(texts, desc="Подготовка текстов", unit="текст"))
    embeddings = compute_embeddings(texts)
    
    print("Шаг 3/3: Сохраняем эмбеддинги в файл...")
    #save_embeddings(df, embeddings, 'embeddings.jsonl')
    
    print("Эмбеддинги сохранены в 'embeddings.jsonl'")

def start_p_2():
    #idea_ids, embeddings = load_embeddings('embeddings.jsonl')
    df_clusters = cluster_embeddings(idea_ids, embeddings, eps=0.25, min_samples=2)
    duplicate_groups, unique_ideas = extract_duplicates_and_uniques(df_clusters)

    print("Найдено повторяющихся групп:", len(duplicate_groups))
    for i, group in enumerate(duplicate_groups, 1):
        print(f"Группа {i}: {group}")

    print("Уникальных идей:", len(unique_ideas))
