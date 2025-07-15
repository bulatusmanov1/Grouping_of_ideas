import pandas as pd
import spacy
import re
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

def extract_duplicates_and_uniques(df_clusters):
    """
    Делит идеи на кластеры повторов и уникальные.
    """
    groups = df_clusters.groupby('cluster_id')['idea_id'].apply(list)

    duplicates = []
    for cid, group in tqdm(groups.items(), desc="Поиск повторов"):
        if cid != -1 and len(group) > 1:
            duplicates.append(group)

    uniques = groups.get(-1, [])
    return duplicates, uniques

def smart_grouping(token_lists, threshold=20):
    """
    Возвращает подгруппы, каждая из которых — список индексов token_lists, схожих по содержанию.
    """
    def similarity(a, b):
        set_a, set_b = set(a), set(b)
        intersection = set_a & set_b
        union = set_a | set_b
        return 100 * len(intersection) / len(union) if union else 0.0

    token_lists = [tokens if tokens else ['АРГЕС'] for tokens in token_lists]
    used = [False] * len(token_lists)
    groups = []

    for i, tokens in enumerate(token_lists):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(token_lists)):
            if not used[j] and similarity(tokens, token_lists[j]) >= threshold:
                group.append(j)
                used[j] = True
        groups.append(group)

    return groups

def get_key_words(texts: List[str]) -> list[list]:
    """
    Извлекает уникальные ключевые слова из текста и возвращает их в список.
    - Полные наименования организаций (например, ПАО "Газпром") как единый элемент.
    - Аббревиатуры (например, ГТС), если они не входят в состав полного наименования организации.
    - Другие ключевые слова по word_pattern.
    - Без дублирования.
    - Исключает отдельные числа, даты, суммы, тарифы и диапазоны.
    - Исключает слова с цифрами, если они заканчиваются на букву в нижнем регистре.
    """
    org_pattern = r'(?:ООО|АО|ПАО|ЗАО|ОАО|ФГБУ|МУП|ГУП|ЧУП|ИП|ГТС ПАО|АВО|СТО)\s*["«][^"»]+?["»]'
    word_pattern = r'\b[А-ЯA-ZЁё0-9][А-Яа-яA-Za-zЁё0-9\-\/\.\"()]{1,}\b'
    date_pattern = r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})'
    number_pattern = r'^\d+([.,]\d+)?$'
    money_pattern = r'\b\d{1,9}(?:[ \u00A0]\d{3})*\s*(?:руб(?:лей|\.|)|р|₽)\b'
    tariff_pattern = r'\b\d{1,9}\s?(?:руб(?:лей|\.|)|р|₽)[/\\][а-яa-z0-9]+'
    range_pattern = r'\b\d{1,4}[-/]\d{1,4}\b'

    key_words_list = []
    for text in texts:
        found = re.findall(word_pattern, text)
        filtered = []
        for w in found:
            if re.fullmatch(date_pattern, w):
                continue
            if re.fullmatch(range_pattern, w):
                continue
            if re.fullmatch(tariff_pattern, w, re.IGNORECASE):
                continue
            if re.fullmatch(money_pattern, w, re.IGNORECASE):
                continue
            if re.fullmatch(number_pattern, w): 
                continue
            if any(c.isdigit() for c in w) and w[-1].islower():
                continue
            if len(w) >= 3 and w.isupper():
                filtered.append(w)
                continue
            if any(c.isdigit() for c in w) or any(c in w for c in '-/()."'):
                filtered.append(w)
                continue
        all_words = filtered
        seen = set()
        unique_filtered = []
        for w in all_words:
            if w not in seen:
                unique_filtered.append(w)
                seen.add(w)
        key_words_list.append(unique_filtered)
    return key_words_list

def get_clean_text(texts: List[str], key_words_list: list[list]) -> list[str]:
    """
    Очищает текст от ключевых слов.
    """
    cleaned_texts = []
    for text, key_words in zip(texts, key_words_list):
        for word in key_words:
            text = re.sub(rf'\b{re.escape(word)}\b', '', text, flags=re.IGNORECASE)
        cleaned_texts.append(text)
    return cleaned_texts