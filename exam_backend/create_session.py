import sqlite3

conn = sqlite3.connect("database/db.sqlite3")
cur = conn.cursor()

cur.execute("""
INSERT INTO sessions (session_id, user_id, start_time, end_time, cheating_score, status)
VALUES ('S001', 'U001', '2025-01-01 10:00', NULL, 10, 'active')
""")

conn.commit()
conn.close()

print("Session S001 created with score = 10")
