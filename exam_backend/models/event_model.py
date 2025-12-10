from sqlalchemy import Column, Integer, String
from database.db import Base

class Event(Base):
    __tablename__ = "events"

    event_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String)
    event_type = Column(String)
    raw_data = Column(String)
    timestamp = Column(String)
