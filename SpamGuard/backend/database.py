# backend/database.py 

import sqlite3
import pandas as pd
import os
import csv

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BACKEND_DIR, 'data') 
DB_PATH = os.path.join(DATA_DIR, "spamguard_feedback.db")
CSV_PATH = os.path.join(DATA_DIR, "2cls_spam_text_cls.csv")

def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            label TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# The rest of the functions in this file use the corrected paths and need no changes.
def add_feedback(message: str, label: str, source: str = 'user'):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO feedback (message, label, source) VALUES (?, ?, ?)", (message, label, source))
    conn.commit()
    conn.close()

def get_feedback_as_df():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT message as Message, label as Category FROM feedback", conn)
    conn.close()
    return df

def get_analytics():
    df_base = pd.read_csv(CSV_PATH, quotechar='"', on_bad_lines='skip')
    base_counts = df_base['Category'].value_counts().to_dict()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT label, COUNT(*) FROM feedback GROUP BY label")
    feedback_counts = dict(cursor.fetchall())
    cursor.execute("SELECT source, COUNT(*) FROM feedback GROUP BY source")
    source_counts = dict(cursor.fetchall())
    conn.close()

    return {
        "base_ham_count": base_counts.get('ham', 0),
        "base_spam_count": base_counts.get('spam', 0),
        "new_ham_count": feedback_counts.get('ham', 0),
        "new_spam_count": feedback_counts.get('spam', 0),
        "user_contribution": source_counts.get('user', 0),
        "llm_contribution": source_counts.get('llm', 0)
    }

def enrich_main_dataset():
    df_feedback = get_feedback_as_df()
    if df_feedback.empty:
        return 0
    records_to_add = df_feedback[['Category', 'Message']].values.tolist()
    try:
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(records_to_add)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM feedback")
        conn.commit()
        conn.close()
        print(f"Successfully enriched dataset with {len(df_feedback)} records.")
        return len(df_feedback)
    except Exception as e:
        print(f"Error during dataset enrichment: {e}")

        raise e
