import psycopg2
import csv
from tqdm import tqdm
from utils.transform import *
from utils.embedding import *
import psycopg2
from db_config import config

conn = psycopg2.connect(**config.DB_SETTINGS)
import psycopg2

class Company_DB:
    def __init__(self, dbname, user, password, host, port):
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        """
        Инициализация соединения с PostgreSQL
        """
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

    def init_db_ideas(self):
        """
        Создание таблицы ideas
        """
        self.cursor.execute('DROP TABLE IF EXISTS ideas;')
        
        self.cursor.execute('''
            CREATE TABLE ideas (
                id SERIAL PRIMARY KEY,
                idea_id TEXT UNIQUE,
                idea_title TEXT,
                idea_description TEXT,
                idea_key_words TEXT[],
                idea_embedding FLOAT8[]
            );
        ''')

    def init_db_clusters(self):
        """
        Создание таблицы clusters
        """
        self.cursor.execute('DROP TABLE IF EXISTS clusters;')

        self.cursor.execute('''
            CREATE TABLE clusters (
                id SERIAL PRIMARY KEY,
                cluster_id TEXT UNIQUE,
                clusters TEXT[]
            );
        ''')

    def insert_data(self, idea_id, idea_title, idea_description, idea_key_words, embedding):
        """
        Вставка записи в таблицу ideas
        """
        self.cursor.execute('''
            INSERT INTO ideas (idea_id, idea_title, idea_description, idea_key_words, idea_embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (idea_id) DO UPDATE SET
                idea_title = EXCLUDED.idea_title,
                idea_description = EXCLUDED.idea_description,
                idea_key_words = EXCLUDED.idea_key_words,
                idea_embedding = EXCLUDED.idea_embedding;
        ''', (idea_id, idea_title, idea_description, idea_key_words, embedding))

    def load_data_from_csv(self, csv_file):
        """
        Загрузка данных из CSV:
        - Чтение всех строк и векторизация батчем
        - Колонки: Номер идеи;Название;Описание
        - Ключевые слова → очистка → эмбеддинг → БД
        """
        ideas = []
        with open(csv_file, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file, delimiter=';')
            for row in tqdm(reader, desc="Чтение CSV"):
                idea_id = row['Номер идеи'].strip()
                idea_title = row['Название'].strip()
                idea_description = row['Описание'].strip()
                combined_text = f"{idea_title} {idea_description}"

                ideas.append({
                    "id": idea_id,
                    "title": idea_title,
                    "description": idea_description,
                    "combined_text": combined_text
                })

        texts = [item["combined_text"] for item in ideas]
        raw_key_words_nested = get_key_words(texts)

        filtered_key_words = []
        for kws in tqdm(raw_key_words_nested, desc="Фильтрация ключевых слов spaCy"):
            filtered = filter_organizations_spacy(kws)
            filtered_key_words.append(filtered)

        cleaned_texts = get_clean_text(texts, filtered_key_words)
        embeddings = compute_embeddings(cleaned_texts)

        for i, idea in enumerate(tqdm(ideas, desc="Загрузка в БД")):
            self.insert_data(
                idea_id=idea["id"],
                idea_title=idea["title"],
                idea_description=idea["description"],
                idea_key_words=filtered_key_words[i],
                embedding=embeddings[i].tolist())
            
    def add_new_ideas(self, list_of_ideas: list[tuple]):
        """
        Добавление новых идей (идея_id, название, описание) вручную:
        - Ключевые слова → очистка → эмбеддинг → БД
        """
        ideas = []
        for idea_id, idea_title, idea_description in list_of_ideas:
            combined_text = f"{idea_title} {idea_description}"
            ideas.append({
                "id": idea_id.strip(),
                "title": idea_title.strip(),
                "description": idea_description.strip(),
                "combined_text": combined_text.strip()
            })

        texts = [item["combined_text"] for item in ideas]
        raw_key_words_nested = get_key_words(texts)

        filtered_key_words = []
        for kws in tqdm(raw_key_words_nested, desc="Фильтрация ключевых слов spaCy"):
            filtered = filter_organizations_spacy(kws)
            filtered_key_words.append(filtered)

        cleaned_texts = get_clean_text(texts, filtered_key_words)
        embeddings = compute_embeddings(cleaned_texts)

        for i, idea in enumerate(tqdm(ideas, desc="Добавление идей в БД")):
            self.insert_data(
                idea_id=idea["id"],
                idea_title=idea["title"],
                idea_description=idea["description"],
                idea_key_words=filtered_key_words[i],
                embedding=embeddings[i].tolist()
            )

    def get_all_ideas(self):
        """
        Получение всех строк из таблицы
        """
        self.cursor.execute('SELECT * FROM users')
        return self.cursor.fetchall()
    
    def process_clusters(self, eps=0.25, min_samples=2, threshold=20):
        """
        Кластеризация идей по эмбеддингам + smart-группировка по ключевым словам
        """
        self.cursor.execute('SELECT idea_id, idea_key_words, idea_embedding FROM ideas;')
        rows = self.cursor.fetchall()

        if not rows:
            print("Нет данных в таблице ideas.")
            return

        idea_ids = []
        key_words = []
        embeddings = []

        for idea_id, kws, emb in rows:
            idea_ids.append(idea_id)
            key_words.append(kws if kws else ['АРГЕС'])
            embeddings.append(emb)

        embeddings = np.array(embeddings)
        df_clusters = cluster_embeddings(idea_ids, embeddings, eps, min_samples)
        duplicate_groups, _ = extract_duplicates_and_uniques(df_clusters)

        total_subgroups = 0
        self.cursor.execute("DELETE FROM clusters;")

        for group_num, group in enumerate(duplicate_groups):
            indices = [idea_ids.index(idea_id) for idea_id in group]
            token_lists = [key_words[i] for i in indices]
            subgroups = smart_grouping(token_lists, threshold)

            for subgroup in subgroups:
                subgroup_ids = [group[i] for i in subgroup]
                self.cursor.execute('''
                    INSERT INTO clusters (cluster_id, clusters)
                    VALUES (%s, %s)
                ''', (f'cluster_{group_num}_{total_subgroups}', subgroup_ids))
                total_subgroups += 1

        print(f"Обработано кластеров: {len(duplicate_groups)}, всего подгрупп: {total_subgroups}")

    def delete_idea(self, idea_id: str):
        """
        Удаление идеи из базы данных по её ID
        Параметры:
            idea_id: идентификатор идеи для удаления
        Возвращает:
            bool: True если удаление прошло успешно, False если идея не найдена
        """
        try:
            self.cursor.execute(
                "DELETE FROM ideas WHERE idea_id = %s",
                (idea_id,)
            )
            if self.cursor.rowcount > 0:
                print(f"Идея {idea_id} успешно удалена")
                self._cleanup_clusters(idea_id)
                return True
            else:
                print(f"Идея {idea_id} не найдена")
                return False
                
        except Exception as e:
            print(f"Ошибка при удалении идеи {idea_id}: {str(e)}")
            return False

    def _cleanup_clusters(self, idea_id: str):
        """
        Приватный метод для очистки упоминаний идеи в кластерах
        """
        try:
            self.cursor.execute(
                "SELECT cluster_id, clusters FROM clusters WHERE %s = ANY(clusters)",
                (idea_id,)
            )
            for cluster_id, clusters in self.cursor.fetchall():
                updated_clusters = [id for id in clusters if id != idea_id]
                if updated_clusters:
                    self.cursor.execute(
                        "UPDATE clusters SET clusters = %s WHERE cluster_id = %s",
                        (updated_clusters, cluster_id)
                    )
                else:
                    self.cursor.execute(
                        "DELETE FROM clusters WHERE cluster_id = %s",
                        (cluster_id,)
                    )   
        except Exception as e:
            print(f"Ошибка при очистке кластеров: {str(e)}")

    def idea_exists(self, idea_id: str) -> bool:
        """
        Проверка наличия идеи по её ID в таблице ideas.
        Возвращает True, если идея существует, иначе False.
        """
        try:
            self.cursor.execute(
                "SELECT 1 FROM ideas WHERE idea_id = %s LIMIT 1;",
                (idea_id,)
            )
            return self.cursor.fetchone() is not None
        except Exception as e:
            print(f"Ошибка при проверке наличия идеи: {str(e)}")
            return False

    def close(self):
        """
        Закрытие соединения
        """
        self.cursor.close()
        self.conn.close()