import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import ingest, retriever
from src.chunker import Chunk


class FakeEmbedder:
    def embed_documents(self, texts):
        vectors = {
            "USB routing information": [1.0, 0.0],
            "Electrical characteristics": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]

    def embed_query(self, query):
        return [1.0, 0.0]


try:
    import chromadb  # noqa: F401

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


@unittest.skipUnless(HAS_CHROMA, "선택 경로(chromadb) 미설치")
class VectorStoreTests(unittest.TestCase):
    def test_persist_and_retrieve_with_similarity_threshold(self) -> None:
        chunks = [
            Chunk(
                text="USB routing information",
                metadata={"source": "datasheet.pdf", "page": 12, "section": "Routing"},
            ),
            Chunk(
                text="Electrical characteristics",
                metadata={"source": "datasheet.pdf", "page": 7, "section": "Electrical"},
            ),
        ]
        # Chroma's Windows HNSW backend can retain a file handle until process exit.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            db_path = Path(directory)
            with (
                patch.object(ingest, "VECTOR_DB_DIR", db_path),
                patch.object(retriever, "VECTOR_DB_DIR", db_path),
                patch.object(ingest, "COLLECTION_NAME", "test_collection"),
                patch.object(retriever, "COLLECTION_NAME", "test_collection"),
            ):
                self.assertEqual(ingest.persist_chunks(chunks, FakeEmbedder()), 2)
                searcher = retriever.ChromaVectorRetriever(
                    embedder=FakeEmbedder(), threshold=0.5
                )
                results = searcher.search("routing", top_k=2)

                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].page, 12)
                self.assertAlmostEqual(results[0].similarity, 1.0)
                del searcher
                gc.collect()


if __name__ == "__main__":
    unittest.main()
