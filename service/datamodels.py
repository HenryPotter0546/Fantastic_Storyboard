# datamodels.py
from sqlalchemy import Column, Integer, String, Date
from .database import Base
import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    credits = Column(Integer, default=10000)
    last_credit_reset_date = Column(Date, default=datetime.date.today)