from sqlalchemy import Column, String, Integer
from database.db import Base

class Session(Base):
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True)
    user_id = Column(String)
    start_time = Column(String)
    end_time = Column(String)
    cheating_score = Column(Integer)
    status = Column(String)
