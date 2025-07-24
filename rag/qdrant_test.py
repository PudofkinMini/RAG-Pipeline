from qdrant_client import QdrantClient, models
import dotenv
import torch
from transformers import AutoTokenizer, AutoModel
from parsing.pdf_document_chunker import PDFDocumentChunker

# Load environment variables from .env file
dotenv.load_dotenv()


# model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

# parser = PDFDocumentChunker("rag\parsing\example.pdf")
# parser.chunk_and_tokenize(tokenizer, overlap_ratio=0.1, chunk_size=512)

# chunk = parser.chunks[0]
# inputs = chunk.get_tokens()
# with torch.no_grad():
#     outputs = model(**inputs)

# # Get embedding (last hidden state)
# embedding = (outputs.last_hidden_state)[
#     :, 0, :
# ]  # shape: [batch_size, sequence_length, hidden_size]

# To get a single vector per sentence, you can use the [CLS] token embedding
qdrant_client = QdrantClient(
    url=dotenv.get_key(".env", "QDRANT_ENDPOINT"),
    api_key=dotenv.get_key(".env", "QDRANT_API_KEY"),
)

# qdrant_client.create_collection(
#     collection_name="Parliament",
#     vectors_config=models.VectorParams(
#         size=embedding.shape[1], distance=models.Distance.COSINE
#     ),
# )

print(qdrant_client.get_collections())
