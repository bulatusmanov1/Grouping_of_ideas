from utils import *
from embedding import *
from transform import *
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
from embedding import match_new_idea_to_old
from utils import load_and_preprocess_data

#test1
app = FastAPI()
app.mount("/static", StaticFiles(directory="pages"), name="static")
templates = Jinja2Templates(directory="pages")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/check_idea/", response_class=HTMLResponse)
async def check_idea(request: Request, title: str = Form(...), description: str = Form(...)):
    df = load_and_preprocess_data('data.csv')
    combined_text = title + " " + description
    results, best_group = match_new_idea_to_old(combined_text, df)
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "results": results, 
        "best_group": best_group,
        "title": title,
        "description": description
    })

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

def step_3():
    idea_ids, contexts, embeddings = json_load('embeddings.jsonl')
    df_clusters = cluster_embeddings(idea_ids, embeddings, eps=0.25, min_samples=2)
    duplicate_groups, _ = extract_duplicates_and_uniques(df_clusters)

    grouped_ideas = []
    total_subgroups = 0

    for group in duplicate_groups:
        group_indices = [idea_ids.index(idea_id) for idea_id in group]
        group_contexts = [contexts[idx] if contexts[idx] else ['АРГЕС'] for idx in group_indices]
        group_ids = [idea_ids[idx] for idx in group_indices]

        subgroups = smart_grouping(group_contexts, threshold=20)

        for subgroup_indices in subgroups:
            subgroup = {
                "idea_ids": [group_ids[idx] for idx in subgroup_indices],
                "texts": [" ".join(group_contexts[idx]) for idx in subgroup_indices]
            }
            grouped_ideas.append(subgroup)
            total_subgroups += 1

    with open("grouped_ideas.json", "w", encoding="utf-8") as f:
        json.dump(grouped_ideas, f, ensure_ascii=False, indent=2)

    print(f"Всего подгрупп: {total_subgroups}")

actions = {
    1: step_1,
    2: step_2,
    3: step_3,
}

def run_steps(steps_to_run):
    for step_num in steps_to_run:
        if step_num in actions:
            print(f"\n==> Running step {step_num}")
            actions[step_num]()
        else:
            print(f"[!] Step {step_num} not found.")

if __name__ == "__main__":
    df = load_and_preprocess_data('data.csv') # Загрузили данные, преобразовали название идеи и её описание в одну ячейку
    #run_steps([3])
    results, best_group = match_new_idea_to_old("Увеличение температуры подачи хим. оч. воды на котлы установок Завода.", df)

    for idea_id, text, sim in results:
        print(f"{idea_id} ({sim}%): {text[:100]}...")

    print("Ближайшая подгруппа включает:", best_group.get('idea_ids', []))


