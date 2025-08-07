import psycopg2
import numpy as np

# Database connection parameters
DB_NAME = "ragdb"
DB_USER = "raguser"
DB_PASSWORD = "ragpassword"
DB_HOST = "localhost"
DB_PORT = "5432"

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
)
cur = conn.cursor()
print(cur)

# Create extension and table
# cur.execute(
#     """
# CREATE EXTENSION IF NOT EXISTS vector;
# CREATE TABLE IF NOT EXISTS chunks (
#     id SERIAL PRIMARY KEY,
#     chunk_id TEXT,
#     text TEXT,
#     metadata JSONB,
#     embedding vector(768) -- adjust dimension as needed
# );
# """
# )
# conn.commit()


# # Function to insert a chunk
# def insert_chunk(chunk_id, text, metadata, embedding):
#     cur.execute(
#         "INSERT INTO chunks (chunk_id, text, metadata, embedding) VALUES (%s, %s, %s, %s)",
#         (chunk_id, text, metadata, embedding.tolist()),
#     )
#     conn.commit()


# # Function to query similar vectors (cosine similarity)
# def query_similar_chunks(query_embedding, top_k=5):
#     cur.execute(
#         """
#         SELECT chunk_id, text, metadata, embedding
#         FROM chunks
#         ORDER BY embedding <=> %s
#         LIMIT %s;
#         """,
#         (query_embedding.tolist(), top_k),
#     )
#     return cur.fetchall()


# # Example usage
# if __name__ == "__main__":
#     # Example embedding (replace with your actual vector)
#     example_embedding = np.random.rand(768).astype(np.float32)
#     insert_chunk("chunk1", "Example text", {"page": 1}, example_embedding)
#     results = query_similar_chunks(example_embedding)
#     for r in results:
#         print(r)

# cur.close()
# conn.close()
