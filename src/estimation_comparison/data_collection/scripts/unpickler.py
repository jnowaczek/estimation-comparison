import pickle
import sqlite3

con = sqlite3.connect("benchmark.sqlite")

con.execute(
    """CREATE TABLE IF NOT EXISTS file_estimations_unpickled(
        file_hash REFERENCES files(file_hash) NOT NULL, 
        preprocessor_id REFERENCES preprocessors(preprocessor_id) NOT NULL,
        estimator_id REFERENCES estimators(estimator_id) NOT NULL, 
        metric REAL
    )""")

con.commit()

rows = con.execute("SELECT * FROM file_estimations").fetchall()

for row in rows:
    unpickled_metric = pickle.loads(row[-1])
    con.execute("INSERT INTO file_estimations_unpickled VALUES(?, ?, ?, ?)",
                (*row[:-1], unpickled_metric if unpickled_metric != "nan" else None))
con.commit()