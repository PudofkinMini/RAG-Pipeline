from abc import ABC, abstractmethod
from typing import List, Dict
import transformers
import torch
import copy


class DocumentChunker(ABC):
    """
    Abstract base class for chunking documents.
    Chunks are segments of a document with contextual meaning (e.g. a section or chapter)
    Chunks are composed of Parts, which are smaller segments of text that
    can each be atomically assigned a location in the document.

    E.g. A Chunk can span multiple pages, each page could be a Part.

    Child classes should implement the `chunk_and_tokenize` method to extract parts &
    chunks from a document.
    """

    class _DocumentChunkPart:
        """
        Represents a part of a document chunk.
        Each part is a segment of text that can be traced back to its
        original location in the document with the `location_id` (e.g. page number).
        """

        def __init__(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            input_type_ids: torch.Tensor,
            original_text: str,
            location_id: str,
        ):
            self.text = None  # Placeholder for the original text
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.token_type_ids = input_type_ids
            self.location_id = location_id

        def copy(self) -> "DocumentChunker._DocumentChunkPart":
            return copy.deepcopy(self)

        def decode(self, tokenizer: transformers.PreTrainedTokenizer) -> str:
            """
            Decodes the tokenized text back to a string.
            """
            return tokenizer.decode(
                self.input_ids[0], skip_special_tokens=True
            )

        def get_length(self) -> int:
            """
            Returns the number of tokens in the part.
            """
            return self.input_ids.shape[1]

    class _DocumentChunk:
        """
        Represents a chunk of a document.
        A DocumentChunk is composed of multiple DocumentChunkParts.
        """

        def __init__(
            self,
            previous_chunk: "DocumentChunker._DocumentChunk" = None,
            overlap_ratio: float = 0,
        ):
            """
            Initialize a DocumentChunk.

            Args:
                previous_chunk (DocumentChunker._DocumentChunk, optional): The previous chunk to inherit parts
                overlap_ratio (float, optional): The ratio of overlap with the previous chunk.
            """
            self.parts = []
            # If there's a previous chunk, we can inherit its parts
            if previous_chunk:
                n_tokens_to_inherit = int(
                    overlap_ratio * previous_chunk.get_length()
                )
                # Go through the parts of the previous chunk in reverse order
                # and add them to the current chunk until we reach the limit
                for part in previous_chunk.parts[::-1]:
                    if (
                        self.get_length() + part.get_length()
                        <= n_tokens_to_inherit
                    ):
                        self.parts = [
                            part,
                        ] + self.parts
                    else:
                        n_tokens_left = n_tokens_to_inherit - self.get_length()
                        self.parts = [
                            DocumentChunker._DocumentChunkPart(
                                input_ids=part.input_ids[
                                    :,
                                    -n_tokens_left:,
                                ],
                                attention_mask=part.attention_mask[
                                    :,
                                    -n_tokens_left:,
                                ],
                                input_type_ids=part.token_type_ids[
                                    :,
                                    -n_tokens_left:,
                                ],
                                location_id=part.location_id,
                                text=part.text,
                            ),
                        ] + self.parts
                        break

        def add_part(self, part: "DocumentChunker._DocumentChunkPart"):
            """
            Adds a DocumentChunkPart to the DocumentChunk.
            """
            self.parts.append(part)

        def get_length(
            self,
        ) -> int:
            """
            Returns the total number of tokens in the chunk.
            """
            return sum(part.get_length() for part in self.parts)

        def copy(self) -> "DocumentChunker._DocumentChunk":
            """
            Returns a copy of the DocumentChunk.
            """
            new_chunk = DocumentChunker._DocumentChunk()
            new_chunk.parts = [part.copy() for part in self.parts]
            return new_chunk

        def decode(self, tokenizer: transformers.PreTrainedTokenizer) -> str:
            """
            Decodes the chunk's parts back to a string.
            """
            return " ".join(part.decode(tokenizer) for part in self.parts)

        def get_tokens(self) -> Dict[str, torch.Tensor]:
            return {
                "input_ids": torch.cat(
                    [part.input_ids for part in self.parts], dim=1
                ),
                "attention_mask": torch.cat(
                    [part.attention_mask for part in self.parts], dim=1
                ),
                "token_type_ids": torch.cat(
                    [part.token_type_ids for part in self.parts], dim=1
                ),
            }

    def __init__(self, document_path: str):
        """
        Initializes the DocumentChunker with the path to the document.

        Args:
            document_path (str): Path to the document file.
        """
        self.document_path = document_path
        self.chunks = []

    @abstractmethod
    def chunk_and_tokenize(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        overlap_ratio: float,
        chunk_size: int,
    ):
        """
        Breaks the document into tokenized chunks.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer
            to use for tokenization.
            overlap_ratio (float): The ratio of overlap between chunks.
            chunk_size (int, optional): The size of each chunk in tokens.
        """
        pass

    def get_tokens(self, chunk_id: int) -> str:
        """
        Returns the full text of a specific chunk by its ID.

        Args:
            chunk_id (int): The ID of the chunk.

        Returns:
            str: The text of the specified chunk.
        """
        chunk = self.chunks[chunk_id]
        # For all parts, get the tokenized text as a string
        return sum([part.tokenized_text for part in chunk.parts])
