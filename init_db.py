from db.db_class import Company_DB
from db_config.config import DB_SETTINGS

db = Company_DB(**DB_SETTINGS)

print("✅ Инициализация базы данных...")

db.init_db_ideas()
db.init_db_clusters()
db.load_data_from_csv("data.csv")
db.process_clusters()

print("✅ Готово!")