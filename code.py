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
        orgs_full = re.findall(org_pattern, text)
        orgs_full = [re.sub(r'["«»]', '', org) for org in orgs_full]
        orgs_full = [re.sub(r'(\b[А-ЯA-ZЁё]{2,})\s+([А-Яа-яЁё])', r'\1\2', org) for org in orgs_full]  
        found = re.findall(word_pattern, text)
        filtered = []
        for w in found:
            if any(w in org for org in orgs_full):
                continue
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
        all_words = filtered + orgs_full
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