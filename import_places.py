import pandas as pd
import mysql.connector
from mysql.connector import Error

# ===========================================
# RAILWAY MYSQL CONFIG
# ===========================================

HOST = "reseau.proxy.rlwy.net"
PORT = 20120
USER = "root"
PASSWORD = "yxoKylMklJQxwMVUMWzFPrtduRjpsuDU"
DATABASE = "railway"

CSV_FILE = "data/cutigo_master_places.csv"

# ===========================================


def mysql_type(dtype):
    dtype = str(dtype)

    if dtype == "int64":
        return "INT"
    elif dtype == "float64":
        return "DOUBLE"
    elif dtype == "bool":
        return "BOOLEAN"
    else:
        return "TEXT"


try:

    print("=" * 60)
    print("Reading CSV...")
    print("=" * 60)

    df = pd.read_csv(CSV_FILE)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    # Tukar NaN -> None
    df = df.where(pd.notnull(df), None)

    # Simpan ID asal sebagai csv_id
    if "id" in df.columns:
        df.rename(columns={"id": "csv_id"}, inplace=True)

    print("\nConnecting to Railway...")

    conn = mysql.connector.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        database=DATABASE,
        autocommit=False
    )

    cursor = conn.cursor()

    print("Connected!")

    # ======================================
    # DROP TABLE
    # ======================================

    cursor.execute("DROP TABLE IF EXISTS places")

    print("Old table removed.")

    # ======================================
    # CREATE TABLE
    # ======================================

    print("Creating new table...")

    cols = []

    # MySQL auto id
    cols.append("id INT AUTO_INCREMENT PRIMARY KEY")

    for col in df.columns:

        sql_type = mysql_type(df[col].dtype)

        cols.append(f"`{col}` {sql_type}")

    create_sql = f"""
    CREATE TABLE places (
        {', '.join(cols)}
    )
    """

    cursor.execute(create_sql)

    print("Table created.")

    # ======================================
    # INSERT
    # ======================================

    print("Preparing data...")

    column_names = ", ".join(f"`{c}`" for c in df.columns)

    placeholders = ", ".join(["%s"] * len(df.columns))

    insert_sql = f"""
    INSERT INTO places ({column_names})
    VALUES ({placeholders})
    """

    batch_size = 500

    total = len(df)

    print("Importing...\n")

    for start in range(0, total, batch_size):

        batch = df.iloc[start:start + batch_size]

        values = [tuple(row) for row in batch.itertuples(index=False, name=None)]

        cursor.executemany(insert_sql, values)

        conn.commit()

        end = min(start + batch_size, total)

        print(f"Imported {end}/{total}")

    cursor.execute("SELECT COUNT(*) FROM places")

    total_rows = cursor.fetchone()[0]

    print("\n" + "=" * 60)
    print("IMPORT SUCCESSFUL")
    print("=" * 60)
    print(f"Rows in MySQL : {total_rows}")

    cursor.close()
    conn.close()

except Error as e:

    print("\nMYSQL ERROR")
    print(e)

except Exception as e:

    print("\nGENERAL ERROR")
    print(e)