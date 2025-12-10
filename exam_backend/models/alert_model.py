from sqlalchemy import Column, Integer, String
from database.db import Base

class Alert(Base):
    __tablename__ = "alerts"

    alert_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String)
    alert_type = Column(String)
    severity = Column(Integer)
    message = Column(String)
    timestamp = Column(String)
