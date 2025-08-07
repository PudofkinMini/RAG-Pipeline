from .document_chunker import DocumentChunker
from typing import List
import PyPDF2

import transformers


class PDFDocumentChunker(DocumentChunker):
    """
    Parser to extract text segments from PDF files.
    Each page's text is returned as a separate string in the list.
    """

    def chunk_and_tokenize(self, tokenizer, overlap_ratio, chunk_size):
        with open(self.document_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            current_chunk = self._DocumentChunk()
            n_tokens_from_last_chunk = int(overlap_ratio * chunk_size)
            for page_number, page in enumerate(reader.pages):
                # Get & tokenize the page text
                page_text = page.extract_text()
                if not page_text:
                    continue
                tokenized_page = tokenizer(page_text, return_tensors="pt")
                input_ids = tokenized_page["input_ids"]
                attention_mask = tokenized_page["attention_mask"]
                token_type_ids = tokenized_page["token_type_ids"]
                current_token_idx = 0
                # While the rest of the page can't fit into the chunk,
                # add the part that can fit, and start a new chunk
                while (
                    current_chunk.get_length()
                    + (input_ids.shape[1] - current_token_idx)
                    > chunk_size
                ):
                    n_tokens_to_add = chunk_size - current_chunk.get_length()
                    current_chunk.add_part(
                        self._DocumentChunkPart(
                            input_ids=input_ids[
                                :,
                                current_token_idx : current_token_idx
                                + n_tokens_to_add,
                            ],
                            attention_mask=attention_mask[
                                :,
                                current_token_idx : current_token_idx
                                + n_tokens_to_add,
                            ],
                            input_type_ids=token_type_ids[
                                :,
                                current_token_idx : current_token_idx
                                + n_tokens_to_add,
                            ],
                            location_id=f"Page {page_number + 1}",
                        )
                    )
                    current_token_idx += n_tokens_to_add
                    self.chunks.append(current_chunk.copy())
                    current_chunk = self._DocumentChunk(
                        current_chunk, overlap_ratio
                    )
                # Add whatever is left of the page to the current chunk
                if current_token_idx < input_ids.shape[1]:
                    current_chunk.add_part(
                        self._DocumentChunkPart(
                            input_ids=input_ids[:, current_token_idx:],
                            attention_mask=attention_mask[
                                :, current_token_idx:
                            ],
                            input_type_ids=token_type_ids[
                                :, current_token_idx:
                            ],
                            location_id=f"Page {page_number + 1}",
                        )
                    )


if __name__ == "__main__":
    # Example usage
    parser = PDFDocumentChunker("example.pdf")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "nlpaueb/legal-bert-base-uncased"
    )
    parser.chunk_and_tokenize(tokenizer, overlap_ratio=0.1, chunk_size=512)
