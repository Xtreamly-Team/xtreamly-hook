import os
from uvicorn import run
from fastapi import FastAPI
from src.policy import execute_policy


from dotenv import load_dotenv
load_dotenv()

app = FastAPI()


@app.get("/")
def heath(): return 'ok'


@app.get("/hedge")
def hedge(user_id: str):
    return execute_policy(user_id)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8001))
    run(app, host="0.0.0.0", port=port)
