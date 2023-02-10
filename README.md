# Query PDF and CSV with LLM

This repo host fastAPI app for querying PDF and CSV file to look for information using natural language.

The query flow for pdf file was implemented as follow:
1. Read text from pdf file
2. Split the text into chunks
3. Encode the chunks into embedding vectors using huggingface GTR-T5 or OpenAI Ada text embedding
4. Upload text chunks and embedding vectors to vector database [Qdrant](https://qdrant.tech/)
5. Get user query text and encode into embedding vector
6. Search vector database for text chunk whose embedding are closest to query embedding based on cosine similarity
7. Get answer from FLAN-T5 or OpenAI GPT-3 based user query text and text chunk

The query flow for pdf file was implemented as follow:
1. Read csv file as pandas df
2. convert each row to a text chunks
3. The rest of the steps are similar to query PDF file

## Setup

1. Clone this repo
    ```
    git clone https://github.com/haizadtarik/queryfile.git
    ```

2. Install dependencies
    ```
    cd queryPDF
    python -m pip install -r requirements.txt
    ```

3. Pull Qdrant image from docker hub
    ```
    docker pull qdrant/qdrant
    ```

4. Run Qdrant base
    ```
    docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
    ```
    
    ---
    **NOTE**

    To use OpenAI embeddings or GPT create `.env` and put your API key there
    ```
    OPENAI_KEY=<OPEN_API_KEY>
    ```
    ---

5. Bring up fastAPI server
    ```
    python server.py
    ```


