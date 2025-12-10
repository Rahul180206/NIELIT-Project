import sqlite3

conn = sqlite3.connect("database/db.sqlite3")
cur = conn.cursor()

print("\n--- SESSIONS TABLE ---")
for row in cur.execute("SELECT * FROM sessions"):
    print(row)

print("\n--- EVENTS TABLE ---")
for row in cur.execute("SELECT * FROM events"):
    print(row)

print("\n--- ALERTS TABLE ---")
for row in cur.execute("SELECT * FROM alerts"):
    print(row)

conn.close()
