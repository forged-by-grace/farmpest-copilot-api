from fastapi import FastAPI, UploadFile
import uvicorn
from dotenv import load_dotenv

from controller.predict_pest_ctr import predict_pest_ctr
from model import Preference


# Load .env variables
load_dotenv()

# Init fastapi
app = FastAPI(title="Farmpest Copilot")


@app.post('/predict_pest/')
async def predict_pest(file: UploadFile, language: str = 'English', location: str = 'Nigeria'):    
    return await predict_pest_ctr(file=file, language=language, location=location)

# start the server
if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        port=8009,
        reload=True,
        host='localhost',
    )
