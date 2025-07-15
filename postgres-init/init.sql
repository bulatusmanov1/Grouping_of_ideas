CREATE TABLE IF NOT EXISTS ideas (
    idea_id SERIAL PRIMARY KEY,
    idea_title TEXT,
    idea_description TEXT,
    idea_embedding FLOAT8[]
);