import sqlite3
import datetime
from flask import g, current_app
from werkzeug.security import generate_password_hash

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    return g.db

def init_db():
    db = sqlite3.connect(current_app.config['DATABASE'])
    db.row_factory = sqlite3.Row
    db.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            full_name     TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role          TEXT DEFAULT 'user',
            status        TEXT DEFAULT 'active',
            staff_id      TEXT DEFAULT '',
            email         TEXT DEFAULT '',
            department    TEXT DEFAULT '',
            last_login    TEXT,
            created_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id              TEXT PRIMARY KEY,
            user_id         INTEGER NOT NULL,
            patient_name    TEXT,
            patient_age     TEXT,
            patient_sex     TEXT,
            image_path      TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence      REAL NOT NULL,
            top5_json       TEXT,
            urgency_level   TEXT DEFAULT 'low',
            notes           TEXT,
            created_at      TEXT NOT NULL,
            is_deleted      INTEGER DEFAULT 0,
            deleted_by      INTEGER DEFAULT NULL,
            deleted_at      TEXT DEFAULT NULL,
            delete_reason   TEXT DEFAULT NULL,
            FOREIGN KEY (user_id)    REFERENCES users(id),
            FOREIGN KEY (deleted_by) REFERENCES users(id)
        );
    ''')

    # Run migrations for existing databases
    existing_user_cols = [r[1] for r in db.execute("PRAGMA table_info(users)").fetchall()]
    for col, definition in [
        ('status',     "TEXT DEFAULT 'active'"),
        ('staff_id',   "TEXT DEFAULT ''"),
        ('email',      "TEXT DEFAULT ''"),
        ('department', "TEXT DEFAULT ''"),
    ]:
        if col not in existing_user_cols:
            db.execute(f"ALTER TABLE users ADD COLUMN {col} {definition}")

    existing_pred_cols = [r[1] for r in db.execute("PRAGMA table_info(predictions)").fetchall()]
    for col, definition in [
        ('is_deleted',    "INTEGER DEFAULT 0"),
        ('deleted_by',    "INTEGER DEFAULT NULL"),
        ('deleted_at',    "TEXT DEFAULT NULL"),
        ('delete_reason', "TEXT DEFAULT NULL"),
    ]:
        if col not in existing_pred_cols:
            db.execute(f"ALTER TABLE predictions ADD COLUMN {col} {definition}")

    db.commit()

    # Seed admin account
    existing = db.execute("SELECT id FROM users WHERE username='admin'").fetchone()
    if not existing:
        db.execute(
            "INSERT INTO users (username,full_name,password_hash,role,status,staff_id,created_at) VALUES (?,?,?,?,?,?,?)",
            ('admin','System Administrator',generate_password_hash('admin123'),
             'admin','active','GCTU-ADMIN-001',datetime.datetime.now().isoformat()))
        db.execute(
            "INSERT INTO users (username,full_name,password_hash,role,status,staff_id,created_at) VALUES (?,?,?,?,?,?,?)",
            ('drkofi','Dr. Kofi Mensah',generate_password_hash('password123'),
             'user','active','GCTU-HW-001',datetime.datetime.now().isoformat()))
    db.commit()
    db.close()
    print("✅ Database initialised.")

def ensure_referrals_table():
    """Create referrals table if it doesn't exist (safe to call multiple times)."""
    import sqlite3 as _sq
    from flask import current_app as _app
    db = _sq.connect(_app.config['DATABASE'])
    db.executescript('''
        CREATE TABLE IF NOT EXISTS referrals (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id    TEXT NOT NULL,
            requester_id INTEGER NOT NULL,
            owner_id     INTEGER NOT NULL,
            reason       TEXT,
            status       TEXT DEFAULT 'pending',
            reviewed_by  INTEGER DEFAULT NULL,
            reviewed_at  TEXT DEFAULT NULL,
            created_at   TEXT NOT NULL,
            FOREIGN KEY (record_id)    REFERENCES predictions(id),
            FOREIGN KEY (requester_id) REFERENCES users(id),
            FOREIGN KEY (owner_id)     REFERENCES users(id)
        );
    ''')
    db.commit()
    db.close()
