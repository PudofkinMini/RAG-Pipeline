from transformers import AutoTokenizer, AutoModel
import torch
from parsing.pdf_document_chunker import PDFDocumentChunker

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

parser = PDFDocumentChunker("example.pdf")
parser.chunk_and_tokenize(tokenizer, overlap_ratio=0.1, chunk_size=512)

# Get model outputs
with torch.no_grad():
    outputs = model(**inputs)

# Get embeddings (last hidden state)
embeddings = (
    outputs.last_hidden_state
)  # shape: [batch_size, sequence_length, hidden_size]

# To get a single vector per sentence, you can use the [CLS] token embedding
sentence_embedding = embeddings[:, 0, :]  # shape: [batch_size, hidden_size]
print(embeddings.shape)
# print(sentence_embedding)
