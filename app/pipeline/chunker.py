from typing import List, Dict, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from app.core.config import settings

class Chunker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL_NAME)
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4")
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on, 
            strip_headers=False
        )

    def _token_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def split_page(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE_TOKENS,
            chunk_overlap=settings.CHUNK_OVERLAP_TOKENS,
            length_function=self._token_len,
            separators=["\n\n", "\n", " ", ""]
        )

        md_docs = self.markdown_splitter.split_text(text)
        split_docs = text_splitter.split_documents(md_docs)

        chunks = []
        for i, split_doc in enumerate(split_docs):
            meta = metadata.copy()
            meta.update(split_doc.metadata)
            chunks.append({
                "text": split_doc.page_content,
                "metadata": meta
            })
        return chunks

chunker = Chunker()
