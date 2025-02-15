#  Copyright (C) 2025 Julian Nowaczek.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

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