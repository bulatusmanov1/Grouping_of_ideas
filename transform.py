import pandas as pd
import spacy
from typing import List, Tuple
from tqdm import tqdm

nlp = spacy.load("ru_core_news_lg")

def filter_organizations_spacy(data: list[str]) -> list[str]:
    """
    Возвращает только те строки из списка, которые spaCy распознаёт как не организации.

    :param data: список строк
    :return: список строк, распознанных как организации
    """
    def is_organization(name: str) -> bool:
        doc = nlp(name)
        return any(ent.label_ == "ORG" for ent in doc.ents)

    return [x for x in data if is_organization(x)]

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