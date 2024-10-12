#  Copyright (C) 2024 Julian Nowaczek.
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
import collections
import logging
import pickle
import sqlite3
from pathlib import Path
from typing import NamedTuple

InputFile = NamedTuple("InputFile", [("hash", str), ("path", str), ("name", str)])
Ratio = NamedTuple("Ratio", [("hash", str), ("algorithm", str), ("ratio", float)])
Metric = NamedTuple("Metric", [("hash", str), ("estimator", str), ("metric", bytes)])

FriendlyRatio = NamedTuple("Ratio", [("file_name", str), ("algorithm", str), ("ratio", float)])
FriendlyMetric = NamedTuple("Metric", [("file_name", str), ("estimator", str), ("metric", any)])


class BenchmarkDatabase:
    def __init__(self, db_path: Path):
        self.con = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS files(
                file_hash TEXT PRIMARY KEY NOT NULL, 
                path TEXT NOT NULL, 
                name TEXT NOT NULL
            )""")
        self.con.commit()
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS compressors(
                compressor_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, 
                name TEXT NOT NULL UNIQUE
            )""")
        self.con.commit()
        # self.cur.execute("CREATE TABLE tag_types(tag_id INT PRIMARY KEY NOT NULL, tag_name TEXT NOT NULL)")
        # self.cur.execute(
        #     "CREATE TABLE file_tags(file_hash REFERENCES files(file_hash), tag_id REFERENCES tag_types(tag_id))")
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS file_ratios(
                file_hash REFERENCES files(file_hash) NOT NULL, 
                compressor_id REFERENCES compressors(compressor_id) NOT NULL, 
                ratio FLOAT
            )""")
        self.con.commit()
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS estimators(
                estimator_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                name TEXT NOT NULL UNIQUE,
                parameters TEXT
            )""")
        self.con.commit()
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS file_estimations(
                file_hash REFERENCES files(file_hash) NOT NULL, 
                estimator_id REFERENCES estimators(estimator_id) NOT NULL, 
                metric BLOB
            )""")
        self.con.commit()

    def update_compressors(self, compressor_list: [str]):
        try:
            compressors = [(c,) for c in compressor_list]
            self.con.executemany("INSERT OR IGNORE INTO compressors(name) VALUES(?)", compressors)
            self.con.commit()
        except sqlite3.Error as e:
            logging.exception(e)

    def update_file(self, file: InputFile):
        try:
            self.con.execute(
                """INSERT INTO files VALUES(:hash, :path, :name)
                 ON CONFLICT(file_hash) 
                     DO UPDATE SET 
                        path = :path, 
                        name = :name""",
                file
            )
            self.con.commit()
        except sqlite3.Error as e:
            logging.exception(e)

    def update_ratio(self, new_ratio: Ratio):
        try:
            self.con.execute(
                """INSERT INTO file_ratios VALUES(:hash, (SELECT compressor_id FROM compressors WHERE name=:compressor_name), :ratio)""",
                new_ratio
            )
            self.con.commit()
        except sqlite3.Error as e:
            logging.exception(e)

    def get_all_files(self):
        return [InputFile(*row) for row in self.con.execute("SELECT file_hash, path, name FROM files").fetchall()]

    @property
    def input_file_count(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    def get_ratios_for_file(self, hash: str):
        ratios = []
        for row in self.con.execute(
                """SELECT file_hash,
                 (SELECT name FROM compressors WHERE file_ratios.compressor_id=compressors.compressor_id),
                 ratio
                 FROM file_ratios WHERE file_hash = ?""",
                (hash,)).fetchall():
            ratios.append(Ratio(*row))
        return ratios

    def get_all_ratios(self):
        ratios = []
        for row in self.con.execute(
                """SELECT 
                   (SELECT name FROM files where file_ratios.file_hash=files.file_hash),
                   (SELECT name FROM compressors WHERE file_ratios.compressor_id=compressors.compressor_id),
                   ratio
                   FROM file_ratios""").fetchall():
            ratios.append(FriendlyRatio(*row, ))
        return ratios

    def update_estimators(self, estimators: {str: any}):
        try:
            estimator_list = []
            for key, value in estimators.items():
                estimator_list.append({"name": key, "parameters": str(value.parameters)})

            self.con.executemany("INSERT OR IGNORE INTO estimators(name, parameters) VALUES(:name, :parameters)",
                                 estimator_list)
            self.con.commit()
        except sqlite3.Error as e:
            logging.exception(e)

    def update_metric(self, new_metric: Metric):
        try:
            self.con.execute(
                """INSERT INTO file_estimations VALUES(:hash, (SELECT estimator_id FROM estimators WHERE name=:estimator_name), :metric)""",
                new_metric
            )
            self.con.commit()
        except sqlite3.Error as e:
            logging.exception(e, new_metric)

    def get_all_metric(self):
        metrics = []
        for row in self.con.execute(
                """SELECT 
                   (SELECT name FROM files where file_estimations.file_hash=files.file_hash),
                   (SELECT name FROM estimators WHERE file_estimations.estimator_id=estimators.estimator_id),
                   metric
                   FROM file_estimations""").fetchall():
            metrics.append(FriendlyMetric(file_name=row[0], estimator=row[1],
                                          metric=row[2] if not isinstance(row[2], bytes) else pickle.loads(row[2])))
        return metrics

    def get_dataframe(self):
        cursor = self.con.cursor()
        cursor.execute("""SELECT (SELECT name FROM files WHERE file_ratios.file_hash = files.file_hash) as filename,
                                 (SELECT name FROM estimators WHERE file_estimations.estimator_id = estimators.estimator_id) as estimator,
                                 (SELECT name FROM compressors WHERE file_ratios.compressor_id = compressors.compressor_id) as compressor,
                                 file_ratios.ratio as ratio,
                                 file_estimations.metric as metric
                          FROM file_ratios
                                 LEFT OUTER JOIN file_estimations on file_ratios.file_hash = file_estimations.file_hash""")
        return cursor.description, cursor.fetchall()
