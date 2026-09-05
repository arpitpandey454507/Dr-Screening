import sqlite3
from pathlib import Path
from datetime import datetime


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "screenings.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """
    Create and return a SQLite database connection.
    """

    connection = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():
    """
    Create the screenings table if it does not already exist.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS screenings (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            patient_name TEXT,
            patient_id TEXT,
            age INTEGER,
            gender TEXT,
            screening_notes TEXT,

            predicted_label TEXT,
            confidence REAL,

            no_dr_probability REAL,
            mild_probability REAL,
            moderate_probability REAL,
            severe_probability REAL,
            proliferative_probability REAL,

            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()

    connection.close()


# ============================================================
# SAVE SCREENING
# ============================================================

def save_screening(
    patient_name,
    patient_id,
    age,
    gender,
    screening_notes,
    predicted_label,
    confidence,
    probabilities
):
    """
    Save one completed screening to the database.

    probabilities should be a dictionary containing:

        No DR
        Mild
        Moderate
        Severe
        Proliferative DR
    """

    connection = get_connection()

    cursor = connection.cursor()

    created_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute(
        """
        INSERT INTO screenings (
            patient_name,
            patient_id,
            age,
            gender,
            screening_notes,

            predicted_label,
            confidence,

            no_dr_probability,
            mild_probability,
            moderate_probability,
            severe_probability,
            proliferative_probability,

            created_at
        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_name,
            patient_id,
            age,
            gender,
            screening_notes,

            predicted_label,
            confidence,

            probabilities.get("No DR", 0.0),
            probabilities.get("Mild", 0.0),
            probabilities.get("Moderate", 0.0),
            probabilities.get("Severe", 0.0),
            probabilities.get("Proliferative DR", 0.0),

            created_at
        )
    )

    screening_id = cursor.lastrowid

    connection.commit()

    connection.close()

    return screening_id


# ============================================================
# GET ALL SCREENINGS
# ============================================================

def get_all_screenings():
    """
    Return all screenings, newest first.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM screenings
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]


# ============================================================
# GET SINGLE SCREENING
# ============================================================

def get_screening(screening_id):
    """
    Return one screening by ID.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM screenings
        WHERE id = ?
        """,
        (screening_id,)
    )

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)


# ============================================================
# DELETE SCREENING
# ============================================================

def delete_screening(screening_id):
    """
    Delete a screening by ID.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM screenings
        WHERE id = ?
        """,
        (screening_id,)
    )

    deleted = cursor.rowcount > 0

    connection.commit()

    connection.close()

    return deleted


# ============================================================
# AUTO INITIALIZATION
# ============================================================

initialize_database()