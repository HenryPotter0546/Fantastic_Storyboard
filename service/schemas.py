# schemas.py
from pydantic import BaseModel
from typing import Optional
import datetime

class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    credits: int
    last_credit_reset_date: datetime.date

    class Config:
        from_attributes = True # SQLAlchemy 2.0 orm_mode

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None