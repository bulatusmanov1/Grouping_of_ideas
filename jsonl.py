import pandas as pd
import numpy as np
import json
import os
import tempfile
from typing import List, Tuple

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

def json_save(
    df: pd.DataFrame,
    context: list | None,
    output_path: str,
    embeddings: np.ndarray | None = None
) -> None:
    """
    Сохраняет JSONL-файл с записями вида:
    {"idea_id": ..., "context": [...]} или {"idea_id": ..., "context": [...], "embedding": [...]}
    
    :param df: DataFrame с колонкой 'idea_id'
    :param context: список контекстов (list[list[str]]) либо None
    :param output_path: путь к файлу
    :param embeddings: np.ndarray с эмбеддингами (опционально)
    """
    records = []

    for i, row in df.iterrows():
        record = {"idea_id": row['idea_id'], "context": context[i] if context else []}
        if embeddings is not None:
            record["embedding"] = embeddings[i].tolist()
        records.append(record)

    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

def json_load(path: str) -> Tuple[List[str], List[list], np.ndarray | None]:
    """
    Загружает idea_id, context и (если есть) embedding из JSONL.
    
    :returns: (idea_ids, contexts, embeddings) — embedding может быть None
    """
    records = pd.read_json(path, lines=True)

    idea_ids = records['idea_id'].tolist()
    contexts = records['context'].tolist()
    embeddings = None

    if 'embedding' in records.columns:
        embeddings = np.vstack(records['embedding'].values)

    return idea_ids, contexts, embeddings

def json_update(
    path: str,
    idea_id: str,
    new_context: list | None = None,
    new_embedding: list | None = None,
    mode: str = 'update'
) -> None:
    """
    Обновляет или добавляет одну запись по idea_id в JSONL-файле.

    :param path: путь к JSONL-файлу
    :param idea_id: уникальный идентификатор записи
    :param new_context: новый контекст (оставить None, чтобы не менять)
    :param new_embedding: новый эмбеддинг (оставить None, чтобы не менять)
    :param mode: 'update' — изменить существующую запись, 'add' — добавить новую, если не найдена
    :raises ValueError: если запись не найдена при update, или уже существует при add
    """
    found = False
    temp_fd, temp_path = tempfile.mkstemp()

    with os.fdopen(temp_fd, 'w', encoding='utf-8') as tmp_file:
        with open(path, 'r', encoding='utf-8') as src_file:
            for line in src_file:
                record = json.loads(line)
                if record.get('idea_id') == idea_id:
                    if mode == 'update':
                        if new_context is not None:
                            record['context'] = new_context
                        if new_embedding is not None:
                            record['embedding'] = new_embedding
                        found = True
                    elif mode == 'add':
                        raise ValueError(f"Запись с idea_id='{idea_id}' уже существует — нельзя добавить повторно.")
                tmp_file.write(json.dumps(record, ensure_ascii=False) + '\n')

        if mode == 'add':
            new_record = {'idea_id': idea_id}
            if new_context is not None:
                new_record['context'] = new_context
            if new_embedding is not None:
                new_record['embedding'] = new_embedding
            tmp_file.write(json.dumps(new_record, ensure_ascii=False) + '\n')
            found = True

    if not found:
        os.remove(temp_path)
        raise ValueError(f"Запись с idea_id='{idea_id}' не найдена для обновления.")

    os.replace(temp_path, path)


    