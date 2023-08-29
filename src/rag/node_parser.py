"""Generate nodes from documents."""
from abc import ABC, abstractmethod
from functools import cached_property, partial
from typing import Callable, Literal, TYPE_CHECKING

import joblib
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from llama_index.schema import BaseNode, Document, TextNode


class MetadataExtractor(ABC):
    @abstractmethod
    def extract_metadata(self, node, doc):
        ...


class FilePathExtractor(MetadataExtractor):
    def extract_metadata(self, node, doc):
        return {"file_path": doc.metadata["file_path"]}


class TextHashExtractor(MetadataExtractor):
    def extract_metadata(self, node, doc):
        return {"text_hash": joblib.hash(node.text)}


extractors = {
    "file_path_extractor": FilePathExtractor,
    "text_hash_extractor": TextHashExtractor,
}


class MetadataPipeline(BaseModel):
    """Pipeline for extracting metadata from a document."""

    extractors: list[str]

    def update_metadata(self, node, doc):
        for extractor in self.extractors:
            extractor_obj = extractors[extractor]()
            metadata = extractor_obj.extract_metadata(node, doc)
            node.metadata.update(metadata)

    def __hash__(self):
        hash_hex = joblib.hash(self.dict())
        return int(hash_hex, 16)


class TikTokenTokenizer(BaseModel):
    """Tokenizer from TikToken."""

    encoding: Literal["gpt2"]

    # model_config = ConfigDict(
    #     arbitrary_types_allowed=True, keep_untouched=[cached_property]
    # )

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    @cached_property
    def encoder(self) -> Callable[[str], list[int]]:
        import tiktoken

        encoding = tiktoken.get_encoding("gpt2")
        return partial(encoding.encode, allowed_special="all")

    def __hash__(self):
        hash_hex = joblib.hash(self.dict())
        return int(hash_hex, 16)


class SentenceTokenizer(BaseModel):
    """Performe sentence-tokenization - i.e. splitting paragraph into sentences."""

    # model_config = ConfigDict(
    #     arbitrary_types_allowed=True, keep_untouched=[cached_property]
    # )

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    type: Literal["tokenizers/punkt"]

    @cached_property
    def chunker(self):
        import nltk
        from llama_index.utils import get_cache_dir

        nltk_data_dir = get_cache_dir()

        # update nltk path for nltk so that it finds the data
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.append(nltk_data_dir)

        try:
            nltk.data.find(self.type)
        except LookupError:
            nltk.download(self.type.split("/")[-1], download_dir=nltk_data_dir)

        return nltk.sent_tokenize

    def __hash__(self):
        hash_hex = joblib.hash(self.dict())
        return int(hash_hex, 16)


class TextChunker(BaseModel):
    """Text chunker with a preference to split paragraphs and complete sentences."""

    # model_config = ConfigDict(
    #     arbitrary_types_allowed=True, keep_untouched=[cached_property]
    # )

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    chunk_size: int = Field(description="The token chunk size for each chunk.", gt=0)

    chunk_overlap: int = Field(
        description="The token overlap between each chunk.", ge=0
    )

    paragraph_separator: str = Field(description="Seperator between paragraphs.")

    sentence_tokenizer: SentenceTokenizer = Field(
        description="The tokenizer to use to tokenize the text.",
    )

    secondary_chunking_regex: str = Field(
        description="Regex to use to split chunks into smaller chunks.",
    )

    tokenizer: TikTokenTokenizer = Field(
        description="The tokenizer to use to tokenize the text.",
    )

    word_seperator: str = Field(description="Seperator for splitting into words")

    @cached_property
    def splitter(self):
        """Return the splitter."""
        from llama_index.text_splitter.sentence_splitter import SentenceSplitter

        return SentenceSplitter(
            separator=self.word_seperator,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            tokenizer=self.tokenizer.encoder,
            paragraph_separator=self.paragraph_separator,
            chunking_tokenizer_fn=self.sentence_tokenizer.chunker,
            secondary_chunking_regex=self.secondary_chunking_regex,
        )

    def run(self, doc: "Document") -> list["TextNode"]:
        """Chunk the document into text chunks."""
        return self.splitter.split_text(doc)

    def __hash__(self):
        hash_hex = joblib.hash(self.dict())
        return int(hash_hex, 16)


class NodeParser(BaseModel):
    """Parses a document into nodes."""

    # model_config = ConfigDict(
    #     arbitrary_types_allowed=True, keep_untouched=[cached_property]
    # )

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)

    text_chunker: TextChunker
    inherit_metadata_from_doc: bool
    construct_prev_next_relations: bool
    metadata_pipeline: MetadataPipeline

    def run(self, documents: list["Document"]) -> list["BaseNode"]:
        """Parse the documents into nodes."""
        import ray

        ray.init(ignore_reinit_error=True)

        return [
            node
            for node in ray.data.from_items(documents)
            .flat_map(self.parse_document)
            .map_batches(lambda x: x, batch_size=100)
            .iter_rows()
        ]

    def parse_document(self, document: "Document") -> list["BaseNode"]:
        """Parse a document into nodes."""
        text_chunks = self.text_chunker.run(document.text)
        nodes = self._build_nodes_from_splits(
            text_chunks=text_chunks,
            document=document,
            include_metadata=self.inherit_metadata_from_doc,
            include_prev_next_rel=self.construct_prev_next_relations,
        )
        return nodes

    def _build_nodes_from_splits(
        self,
        text_chunks: list[str],
        document: "Document",
        include_metadata: bool = True,
        include_prev_next_rel: bool = False,
    ) -> list["TextNode"]:
        """Build nodes from text chunks."""
        from llama_index.schema import NodeRelationship, TextNode

        nodes: list[TextNode] = []
        for i, text_chunk in enumerate(text_chunks):
            node_metadata = {}
            if include_metadata:
                node_metadata = document.metadata

            node = TextNode(
                text=text_chunk,
                embedding=document.embedding,
                metadata=node_metadata,
                excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                metadata_seperator=document.metadata_seperator,
                metadata_template=document.metadata_template,
                text_template=document.text_template,
                relationships={
                    NodeRelationship.SOURCE: document.as_related_node_info()
                },
            )
            self.metadata_pipeline.update_metadata(node=node, doc=document)
            nodes.append(node)

        if include_prev_next_rel:
            for i, node in enumerate(nodes):
                if i > 0:
                    node.relationships[NodeRelationship.PREVIOUS] = nodes[
                        i - 1
                    ].as_related_node_info()
                if i < len(nodes) - 1:
                    node.relationships[NodeRelationship.NEXT] = nodes[
                        i + 1
                    ].as_related_node_info()

        return nodes

    def __hash__(self):
        hash_hex = joblib.hash(self.dict())
        return int(hash_hex, 16)


def parse_document(self, document: "Document") -> list["BaseNode"]:
    """Parse a document into nodes."""
    text_chunks = self.text_chunker.run(document.text)
    nodes = self._build_nodes_from_splits(
        text_chunks=text_chunks,
        document=document,
        include_metadata=self.inherit_metadata_from_doc,
        include_prev_next_rel=self.construct_prev_next_relations,
    )
    return nodes
