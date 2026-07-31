from pathlib import Path

from src.utils.vector_db import FAISSVectorDB


db_dir = Path("C:/nlptemplate_decoder_rag/vector_db")

vector_db = FAISSVectorDB.load(
    directory=db_dir,
    embedding_dim=32,  # Укажите вашу размерность
    index_type="flat",  # <--- Изменили с 'IndexFlatIP' на 'flat'
    normalize_embeddings=True,
)

print(f"Всего векторов в базе: {vector_db.index.ntotal}")

# Проверяем метаданные первых 5 документов
for i in range(min(5, len(vector_db.metadata))):
    print(f"Документ {i}:", vector_db.metadata[i])
