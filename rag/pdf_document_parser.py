from document_chunker import DocumentChunker
from typing import List
import PyPDF2

import transformers


class PDFDocumentParser(DocumentChunker):
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
                tokenized_page_text = tokenizer(
                    page_text, return_tensors="pt"
                )["input_ids"]
                current_token_idx = 0
                # While the rest of the page can't fit into the chunk,
                # add the part that can fit, and start a new chunk
                print(
                    "\n",
                    f"New Page: {page_number + 1} | ",
                    f"Length of Current Chunk: {current_chunk.get_length()} | ",
                    f"Tokens on this page: {tokenized_page_text.shape[1]}",
                )
                while (
                    current_chunk.get_length()
                    + (tokenized_page_text.shape[1] - current_token_idx)
                    > chunk_size
                ):
                    n_tokens_to_add = chunk_size - current_chunk.get_length()
                    print(
                        f"Filling Chunk, adding {n_tokens_to_add} tokens to {current_chunk.get_length()}"
                    )
                    current_chunk.add_part(
                        self._DocumentChunkPart(
                            tokenized_text=tokenized_page_text[
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
                    print(
                        f"New chunk created of size {current_chunk.get_length()}"
                    )
                # Add whatever is left of the page to the current chunk
                if current_token_idx < tokenized_page_text.shape[1]:
                    print(
                        f"Adding remaining {tokenized_page_text.shape[1] - current_token_idx} tokens to current chunk of size {current_chunk.get_length()}"
                    )
                    current_chunk.add_part(
                        self._DocumentChunkPart(
                            tokenized_text=tokenized_page_text[
                                :, current_token_idx:
                            ],
                            location_id=f"Page {page_number + 1}",
                        )
                    )


if __name__ == "__main__":
    # Example usage
    parser = PDFDocumentParser("example.pdf")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "nlpaueb/legal-bert-base-uncased"
    )
    parser.chunk_and_tokenize(tokenizer, overlap_ratio=0.1, chunk_size=512)

    for chunk in parser.chunks[:3]:
        print(chunk.decode(tokenizer))
        print("")
