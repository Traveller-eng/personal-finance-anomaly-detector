"""
database.py — Local State & Feedback Persistence Layer
======================================================
Stores user feedback (expected transactions), budget goals, and persisted
machine learning model objects using SQLite.
"""
import sqlite3
import pickle
import os

def get_db_path():
    return os.environ.get('PFAD_TEST_DB', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pfad.db'))

DB_PATH = get_db_path() # Fallback for backward compatibility, though not dynamically updated

def init_db():
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    # Ignoring known / expected false-positive anomalies
    c.execute('''
        CREATE TABLE IF NOT EXISTS expected_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant TEXT,
            amount_min REAL,
            amount_max REAL
        )
    ''')
    # Monthly spending caps
    c.execute('''
        CREATE TABLE IF NOT EXISTS budgets (
            category TEXT PRIMARY KEY,
            monthly_limit REAL
        )
    ''')
    # Serialized Isolation Forest model persistence
    c.execute('''
        CREATE TABLE IF NOT EXISTS model_snapshots (
            id INTEGER PRIMARY KEY,
            model_blob BLOB,
            data_hash TEXT,
            contamination REAL
        )
    ''')
    conn.commit()
    conn.close()

def add_expected_transaction(merchant: str, amount: float, tolerance: float = 0.15):
    """
    Learns a rule to ignore an anomaly. Defaults to +/- 15% tolerance.
    """
    val_min = amount * (1.0 - tolerance)
    val_max = amount * (1.0 + tolerance)
    
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("INSERT INTO expected_transactions (merchant, amount_min, amount_max) VALUES (?, ?, ?)",
              (merchant, val_min, val_max))
    conn.commit()
    conn.close()

def get_expected_transactions():
    """Returns a list of dicts mapping merchants to safe amount bands."""
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT merchant, amount_min, amount_max FROM expected_transactions")
    res = c.fetchall()
    conn.close()
    return [{"merchant": r[0].lower(), "amount_min": r[1], "amount_max": r[2]} for r in res]

def set_budget(category: str, limit: float):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO budgets (category, monthly_limit) VALUES (?, ?)", (category.lower().strip(), float(limit)))
    conn.commit()
    conn.close()

def get_budgets() -> dict:
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT category, monthly_limit FROM budgets")
    res = c.fetchall()
    conn.close()
    return {r[0]: r[1] for r in res}

def save_model(model, data_hash: str, contamination: float):
    """Pickles the sklearn model into SQLite for rapid reloading."""
    blob = pickle.dumps(model)
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("DELETE FROM model_snapshots")  # Restrict to standard active model 
    c.execute("INSERT INTO model_snapshots (id, model_blob, data_hash, contamination) VALUES (1, ?, ?, ?)",
              (sqlite3.Binary(blob), data_hash, contamination))
    conn.commit()
    conn.close()

def load_model(data_hash: str, contamination: float):
    """
    Attempts to retrieve a serialized model. Only returns the model if the
    data hash and contamination config haven't changed.
    """
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT model_blob FROM model_snapshots WHERE data_hash = ? AND contamination = ?", (data_hash, contamination))
    res = c.fetchone()
    conn.close()
    if res:
        return pickle.loads(res[0])
    return None

# Auto-initialize tables when imported
init_db()
