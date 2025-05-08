from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import build_qa_chain
import uvicorn

app = FastAPI()
qa_chain = build_qa_chain()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    response = qa_chain.invoke({"input": query.question})
    return {"answer": response["answer"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)