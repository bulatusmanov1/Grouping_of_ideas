from utils import *
from embedding import *
from transform import *

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
    duplicate_groups, _ = extract_duplicates_and_uniques(df_clusters)

    total_subgroups = 0

    for i, group in enumerate(duplicate_groups, 1):
        #print(f"\n=== Кластер #{i} (всего {len(group)} идей) ===")

        group_indices = [idea_ids.index(idea_id) for idea_id in group]
        group_contexts = [contexts[idx] if contexts[idx] else ['АРГЕС'] for idx in group_indices]
        group_ids = [idea_ids[idx] for idx in group_indices]

        subgroups = smart_grouping(group_contexts, threshold=20)

        for j, subgroup_indices in enumerate(subgroups, 1):
            #print(f"  Подгруппа {j}:")
            for idx in subgroup_indices:
                #print(f"    - {group_ids[idx]}")
                pass
            total_subgroups += 1

    #print(f"\nВсего подгрупп: {total_subgroups}")
    from collections import defaultdict

    cluster_subgroups_contexts = defaultdict(list)

    for group in duplicate_groups:
        group_indices = [idea_ids.index(idea_id) for idea_id in group]
        group_contexts = [contexts[idx] if contexts[idx] else ['АРГЕС'] for idx in group_indices]

        subgroups = smart_grouping(group_contexts, threshold=20)
        for subgroup in subgroups:
            subgroup_contexts = [group_contexts[i] for i in subgroup]
            cluster_subgroups_contexts[len(cluster_subgroups_contexts)].append(subgroup_contexts)

    print(cluster_subgroups_contexts)

    #print(f"[debug] Кол-во подгрупп в cluster_subgroups_contexts: {len(cluster_subgroups_contexts)}")
    #find_best_subgroup_for_new_idea('Вывод из эксплуатации печи П-1 на установке №7;Предлагается вывести из эксплуатации печь П-1 на установке № 7 за счёт  монтажа дополнительного теплообменника для подогрева сырья в колонну К-1.', cluster_subgroups_contexts)



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