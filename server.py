import uvicorn
from fastapi import FastAPI, File, UploadFile
from queryPDF import queryVDB, uploadPDF

app = FastAPI(
    title="Query PDF using LLM",
    description="Use LLM to search for information in PDF files",
    version="0.1",
)

@app.get("/", tags=['Home'], name='home')
async def root():
    return {"message": "QueryPDF v0.1. Go to <BASE_URL>/docs to see the API documentation."}

@app.post("/upload/", tags=['Upload'], name='Upload pdf file and create a new collection')
def create_upload_file(file: UploadFile = File(...)):
    uploadPDF(file)
    file.file.close()
    return {"results": "Collection created successfully"}

@app.get("/query/{vdb_name}/{user_query}", tags=['Query'], name='Search for information in PDF file')
def search_pdf(vdb_name: str, user_query: str):
    query = queryVDB(vdb_name)
    results = query.llm_reply(user_query)
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080)


