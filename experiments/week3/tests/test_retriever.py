import unittest
import tempfile
import os
from retriever import Retriever


class TestRetriever(unittest.TestCase):
    def setUp(self):
        # make two tiny docs on disk
        self.paths = []
        for txt in ["foo bar baz", "lorem ipsum dolor sit amet"]:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            tf.write(txt.encode("utf-8"))
            tf.close()
            self.paths.append(tf.name)

    def tearDown(self):
        for p in self.paths:
            os.unlink(p)

    def test_add_and_query(self):
        r = Retriever(chunk_size=20, chunk_overlap=5)
        r.add_documents(self.paths)

        # query something in doc2
        hits = r.query("ipsum", top_k=1)
        self.assertTrue(any("ipsum" in chunk for chunk, _ in hits))


if __name__ == "__main__":
    unittest.main()
