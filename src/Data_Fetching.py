#!/usr/bin/python

import sqlite3
import pandas as pd
import os.path

def fetch(database_path):

    PROJECT_PATH = os.path.realpath(os.path.dirname(__file__))
    filepath = os.path.join(PROJECT_PATH, database_path)

    conn = sqlite3.connect(filepath)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM id_text")
    text = cursor.fetchall()
    text_df = pd.DataFrame(text, columns=['id', 'tweet'])

    cursor.execute("SELECT * FROM id_dialect")
    dialect = cursor.fetchall()
    dialect_df = pd.DataFrame(dialect, columns=['id', 'dialect'])

    cursor.close()
    conn.close()

    df = pd.merge(text_df, dialect_df, on="id")
    df.drop(['id'], axis=1, inplace=True)

    return df
