from utils import *
from embedding import *
from transform import *

"""
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
"""
def step_1():
    '''
    Достали из общей ячейки список ключевых слов, удалили из них названия, создали слова 
    '''
    list_context = get_key_words(df['full_text'].tolist())
    list_context_update = [filter_organizations_spacy(i) for i in tqdm(list_context, desc="Фильтрация организаций")]
    final_text = get_clean_text(df['full_text'].tolist(), list_context_update)
    embeddings = compute_embeddings(list(tqdm(final_text, desc="Подготовка текстов", unit="текст")))
    json_save(df[['idea_id']], list_context_update, 'embeddings.jsonl', embeddings)

def step_2():
    idea_ids, contexts, embeddings = json_load('embeddings.jsonl')
    df_clusters = cluster_embeddings(idea_ids, embeddings, eps=0.25, min_samples=2)
    duplicate_groups, unique_ideas = extract_duplicates_and_uniques(df_clusters)
    k = 0  # общее число подгрупп
    c = 0  # пока не используется

    for i, group in enumerate(duplicate_groups, 1):
        print(f"\n=== Обрабатываем кластер #{i} с {len(group)} идеями ===")
        group_indices = [idea_ids.index(idea_id) for idea_id in group]
        group_contexts = [contexts[idx] for idx in group_indices]
        group_ids = [idea_ids[idx] for idx in group_indices]
        subgroups = smart_grouping(group_contexts, threshold=20)

        for j, subgroup in enumerate(subgroups, 1):
            k += 1
            print(f"  Подгруппа {j}: {len(subgroup)} элементов")
            for context in subgroup:
                idx = group_contexts.index(context)
                idea_id = group_ids[idx]
                print("   -", idea_id)

    print(f"\nВсего подгрупп: {k}, пропущено: {c}")
    

actions = {
    1: step_1,
    2: step_2,
}

def run_steps(steps_to_run):
    for step_num in steps_to_run:
        if step_num in actions:
            print(f"\n==> Running step {step_num}")
            actions[step_num]()
        else:
            print(f"[!] Step {step_num} not found.")

if __name__ == "__main__":
    #df = load_and_preprocess_data('data.csv') # Загрузили данные, преобразовали название идеи и её описание в одну ячейку
    run_steps([2])