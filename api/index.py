from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from api.inget import process_query

app = FastAPI()

# Allow CORS for the React frontend
origins = ["http://localhost:3000", "http://localhost:5173",
           "http://127.0.0.1:5173"]  # Update with your frontend URL

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/prompts")
async def get_prompt_response(msg: str = Query(...)):
    # Process the message and generate a response
    response = await process_query(msg)
    return {"response": response}
