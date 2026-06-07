# landmarks_db.py

import json
import sqlite3
import pandas as pd


def store_landmarks_in_db(landmarks_data, db_path='../landmarks.db'):
    """
    Stores landmark extraction results into a SQLite database.
    The landmarks field is stored as a JSON string.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Create the table if it doesn't exist.
    c.execute('''
        CREATE TABLE IF NOT EXISTS landmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT,
            split TEXT,
            label TEXT,
            landmarks TEXT
        )
    ''')
    # Insert each record.
    for item in landmarks_data:
        landmarks_json = json.dumps(item['landmarks']) if item['landmarks'] is not None else None
        c.execute('''
            INSERT INTO landmarks (file_path, split, label, landmarks)
            VALUES (?, ?, ?, ?)
        ''', (item['file_path'], item['split'], item['label'], landmarks_json))
    conn.commit()
    conn.close()

