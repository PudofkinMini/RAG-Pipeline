from qdrant_client import QdrantClient, models
import dotenv
import torch
from transformers import AutoTokenizer, AutoModel
from parsing.pdf_document_chunker import PDFDocumentChunker
from typing import List, Dict

# Load environment variables from .env file
dotenv.load_dotenv()


def upsert_embeddings(
    collection_name: str,
    document_source: str,
    embeddings: List[torch.Tensor],
    chunks: List[PDFDocumentChunker._DocumentChunk],
):
    """
    Upserts embeddings into the Qdrant collection.

    Args:
        collection_name (str): Name of the Qdrant collection.
        document_source (str): Name of the source document (e.g., file path).
        embeddings (List[torch.Tensor]): List containing the embeddings.
        chunks (List[PDFDocumentChunker._DocumentChunk]): List of document chunks
        that generated the embeddings.
    """
    points = []
    for i, chunk in enumerate(chunks):
        inputs = chunk.get_tokens()
        with torch.no_grad():
            outputs = model(**inputs)

        # Get embedding (last hidden state)
        embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        points.append(
            models.PointStruct(
                id=i + 1,
                payload={"location": chunk.decode(tokenizer)},
                vector=embedding.squeeze().tolist(),
            )
        )

    qdrant_client.upsert(collection_name=collection_name, points=points)


# To get a single vector per sentence, you can use the [CLS] token embedding
qdrant_client = QdrantClient(
    url=dotenv.get_key(".env", "QDRANT_ENDPOINT"),
    api_key=dotenv.get_key(".env", "QDRANT_API_KEY"),
)

model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

chunker = PDFDocumentChunker("rag\parsing\example.pdf")
chunker.chunk_and_tokenize(tokenizer, overlap_ratio=0.1, chunk_size=512)

embeddings = []

for chunk in chunker.chunks:
    inputs = chunk.get_tokens()
    with torch.no_grad():
        outputs = model(**inputs)

    # Get embedding (last hidden state)
    embedding = (outputs.last_hidden_state)[
        :, 0, :
    ]  # shape: [batch_size, sequence_length, hidden_size]

    embeddings.append(embedding)

# Upsert the points
qdrant_client.upsert(
    collection_name="{collection_name}",
    points=[
        models.PointStruct(
            id=1,
            payload={
                "color": "red",
            },
            vector=[0.9, 0.1, 0.1],
        ),
        models.PointStruct(
            id=2,
            payload={
                "color": "green",
            },
            vector=[0.1, 0.9, 0.1],
        ),
        models.PointStruct(
            id=3,
            payload={
                "color": "blue",
            },
            vector=[0.1, 0.1, 0.9],
        ),
    ],
)
