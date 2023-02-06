from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class uploadPDF:
    def __init__(self, uploaded_file, password=None, encoder='sentence-transformers/gtr-t5-base'):
        self.model = SentenceTransformer(encoder)
        self.qdrant_client = QdrantClient(host='localhost', port=6333)
        self.embeddings, self.chunks_dict = self.parse2doc(uploaded_file,password)
        self.create_collection()
        self.upload_vectors(self.embeddings,self.chunks_dict)

    def parse2doc(self, uploaded_file, password=None):
        """
        This function is used to parse pdf file to documents
        
        Returns:
        embeddings (list): The list of embeddings vectors
        chunks_dict (list): The list of text chunks in dictionary format
        """
        if type(uploaded_file) == str:
            self.collection_name = uploaded_file.split('.')[0]
            reader = PdfReader(uploaded_file,password=password)
        else:
            self.collection_name = uploaded_file.filename.split('.')[0]
            reader = PdfReader(uploaded_file.file,password=password)
        pages = reader.pages
        chunks = []
        for page in pages:
            text = page.extract_text()
            chunks.append(text)
        embeddings = self.model.encode(chunks)
        chunks_dict = [{'text':chunk} for chunk in chunks]
        return embeddings, chunks_dict


    def create_collection(self,distance_metric=Distance.COSINE):
        """
        This function is used to create a collection in Qdrant vector database
        
        Parameters:
        distance_metric (str): The distance metric used to calculate distance between vectors
        """
        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.model.get_sentence_embedding_dimension(), distance=distance_metric),
        )

    def upload_vectors(self,vectors,payload):
        """
        This function is used to upload vectors to Qdrant vector database
        
        Parameters:
        vectors (list): The list of embeddings vectors
        payload (list): The list of text chunks in dictionary format
        """
        self.qdrant_client.upload_collection(
            collection_name=self.collection_name, 
            vectors=vectors,
            payload=payload,
            ids=None,  
            batch_size=8  
        )

class queryVDB:
    def __init__(self, collection_name, encoder='sentence-transformers/gtr-t5-base', llm='google/flan-t5-base'):
        self.collection_name = collection_name
        self.encoder = SentenceTransformer(encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(llm)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(llm)
        self.qdrant_client = QdrantClient(host='localhost', port=6333)
        

    def search_vectors(self,query_vector,limit=3):  
        """
        This function is used to search vectors in Qdrant vector database
        
        Parameters:
        query_vector (list): The list of embeddings vectors
        limit (int): The number of results to be returned
        
        Returns:
        results (list): The first item of list of results
        """
        results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                with_vectors=True,
                with_payload=True,
                limit=limit
            )
        return results[0]

    def search_text(self,query_text,limit=3):
        """
        This function is used to search pdf file for information

        Parameters:
        query_text (str): The text to be searched

        Returns:
        results (dict): The matched chunk of text with query text
        """
        query_vector = self.encoder.encode([query_text])
        results = self.search_vectors(query_vector[0],limit=limit)
        return results.payload
    
    def llm_reply(self, query_text):
        """
        This function is used to search pdf file for information

        Parameters:
        query_text (str): The text to be searched

        Returns:
        results (dict): The matched chunk of text with query text
        """
        document = self.search_text(query_text)

        t5query = f"Question: {query_text}. Context: {document}"
        inputs = self.tokenizer(t5query, truncation=True, max_length=512, return_tensors="pt")
        outputs = self.llm.generate(**inputs, max_new_tokens=512)
        result =  self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return result

if __name__ == '__main__':
    uploadPDF('eCCRIS_FAQ.pdf')
    query = queryVDB('eCCRIS_FAQ')
    results = query.llm_reply("How to apply for eCCRIS?")
    print(results)

