from fastapi import FastAPI

import learn 

app = FastAPI()

@app.get("/item")
def read_item(hidden: str, age: int, city: str):
    return {"name": name, "age": age, "city": city}

