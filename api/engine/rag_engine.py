from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.readers.json import JSONReader

class RagEngine():
    def __init__(self):
        self.top_k = 3
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = None
        Settings.chunk_size = 256
        Settings.chunk_overlap = 25
        self.reader = JSONReader()
        self.documents = self.reader.load_data(input_file="data.json", extra_info={})
        self.index = VectorStoreIndex.from_documents(self.documents)
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=3,
        )

    def create_context_processor(self,prompt:str):


        query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
        )
        response = query_engine.query(prompt)
        # reformat response
        context = "Context:\n"
        for i in range(self.top_k):
            context = context + response.source_nodes[i].text + "\n\n"
        return context


if __name__ == '__main__':
    rag = RagEngine()
    context = rag.create_context_processor("Salad: Burrata & Confit Cherry Tomatoes House-made burrata with confit cherry tomatoes,")
    print(context)