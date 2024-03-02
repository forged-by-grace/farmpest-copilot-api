from pydantic import BaseModel, Field
from fastapi import UploadFile


class Preference(BaseModel):
    language: str = Field(default='English', description='The client language.')
    location: str = Field(default='Nigeria', description='The client location')