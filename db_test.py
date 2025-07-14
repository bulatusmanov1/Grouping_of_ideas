import sqlite3
import csv

class Company_DB:
    def __init__(self, db_file=":memory:"):
        """
        Инициализация соединения с БД в памяти
        """
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()

    def init_db(self):
        """
        Инициализация таблицы в БД
        """
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea_id TEXT UNIQUE,
            idea_title TEXT,
            idea_description TEXT,
            idea_full_text TEXT,
            idea_key_words TEXT,
            idea_embedding BLOB
        );
        ''')
        self.conn.commit()

    def insert_data(self, idea_id, idea_title = "", idea_description = "", idea_full_text = "", idea_key_words = "", embedding = ""):
        """
        Вставка данных о новой идее в таблицу
        """
        self.cursor.execute('''
        INSERT OR REPLACE INTO users (idea_id, idea_title, idea_description, idea_full_text, idea_key_words, idea_embedding)
        VALUES (?, ?, ?, ?, ?, ?);
        ''', (idea_id, idea_title, idea_description, idea_full_text, idea_key_words, embedding))
        self.conn.commit()

    def get_all_ideas(self):
        """
        Получить все идеи из базы данных
        """
        self.cursor.execute('SELECT * FROM users')
        return self.cursor.fetchall()

    def get_embedding(self, idea_id):
        """
        Получить эмбеддинг по ID идеи
        """
        self.cursor.execute('SELECT idea_title FROM users WHERE idea_id = ?', (idea_id,))
        result = self.cursor.fetchone()
        return result

    def close(self):
        """
        Закрытие соединения с БД
        """
        self.conn.close()
