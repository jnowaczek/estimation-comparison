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
import hashlib
import logging
import os
import pickle
import sqlite3
from pathlib import Path
from timeit import default_timer
from typing import Tuple, List

import dask.distributed

from estimation_comparison.model import InputFile, Ratio, FriendlyRatio, FriendlyMetric, Compressor, Estimator, \
    Preprocessor, Result


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
                name TEXT NOT NULL UNIQUE,
                parameters BLOB
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
                parameters BLOB
            )""")
        self.con.commit()
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS file_estimations(
                file_hash REFERENCES files(file_hash) NOT NULL, 
                preprocessor_id REFERENCES preprocessors(preprocessor_id) NOT NULL,
                estimator_id REFERENCES estimators(estimator_id) NOT NULL, 
                metric BLOB
            )""")
        self.con.commit()
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS preprocessors(
                preprocessor_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                name TEXT NOT NULL UNIQUE,
                parameters BLOB
            )""")
        self.con.commit()

    def update_compressors(self, compressors: List[Compressor]):
        try:
            compressor_list = []
            for c in compressors:
                compressor_list.append(
                    {"name": c.name, "parameters": pickle.dumps(c.instance.traits(), protocol=pickle.HIGHEST_PROTOCOL)})

            self.con.executemany("INSERT OR IGNORE INTO compressors(name, parameters) VALUES(:name, :parameters)",
                                 compressor_list)
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

    def get_ratios_for_file(self, file_hash: str):
        ratios = []
        for row in self.con.execute(
                """SELECT file_hash,
                 (SELECT name FROM compressors WHERE file_ratios.compressor_id=compressors.compressor_id),
                 ratio
                 FROM file_ratios WHERE file_hash = ?""",
                (file_hash,)).fetchall():
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

    def update_estimators(self, estimators: List[Estimator]):
        try:
            estimator_list = []
            for e in estimators:
                estimator_list.append(
                    {"name": e.name, "parameters": pickle.dumps(e.instance.traits(), protocol=pickle.HIGHEST_PROTOCOL)})

            self.con.executemany("INSERT OR IGNORE INTO estimators(name, parameters) VALUES(:name, :parameters)",
                                 estimator_list)
            self.con.commit()
        except sqlite3.Error as e:
            logging.exception(e)

    def update_preprocessors(self, preprocessors: List[Preprocessor]):
        try:
            preprocessor_list = []
            for p in preprocessors:
                preprocessor_list.append(
                    {"name": p.name, "parameters": pickle.dumps(p.instance.traits(), protocol=pickle.HIGHEST_PROTOCOL)})

            self.con.executemany("INSERT OR IGNORE INTO preprocessors(name, parameters) VALUES(:name, :parameters)",
                                 preprocessor_list)
            self.con.commit()
        except sqlite3.Error as e:
            logging.exception(e)

    def update_result(self, result: Result):
        try:
            self.con.execute(
                """INSERT INTO file_estimations VALUES(?, 
                                                          (SELECT preprocessor_id FROM preprocessors WHERE name=?), 
                                                          (SELECT estimator_id FROM estimators WHERE name=?),
                                                          ?)""",
                (result.input_file.hash, result.preprocessor.name, result.estimator.name,
                 pickle.dumps(result.value, protocol=pickle.HIGHEST_PROTOCOL))
            )
            self.con.commit()
        except sqlite3.Error as e:
            logging.exception(e, result)

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

    @staticmethod
    def _hash_file(p: Path) -> str:
        with open(p, "rb") as f:
            # noinspection PyTypeChecker
            return hashlib.file_digest(f, hashlib.sha256).hexdigest()

    def update_files(self, client: dask.distributed.Client, locations):
        hash_tasks = []

        # Glob 'em, hash 'em, and INSERT 'em
        for s in locations:
            path = Path(s)
            logging.debug(f"Entering directory '{path}'")
            for file in filter(lambda f: f.is_file(), path.glob("**/*")):
                future = client.submit(self._hash_file, file)
                future.context = (str(file), os.path.relpath(file, path))
                hash_tasks.append(future)

        for future, result in dask.distributed.as_completed(hash_tasks, with_results=True):
            self.update_file(InputFile(result, future.context[0], future.context[1]))

    @staticmethod
    def _ratio_file(compressor: Compressor, f: InputFile) -> Tuple[InputFile, Compressor, float]:
        try:
            with open(f.path, "rb") as fd:
                return f, compressor, compressor.instance.run(fd.read())
        except ValueError as e:
            logging.warning(e)

    def update_ratios(self, client: dask.distributed.Client, compressors: List[Compressor]):
        ratio_start_time = default_timer()
        ratio_tasks = []
        submitted_ratio_tasks = 0
        completed_ratio_tasks = 0

        for f in self.get_all_files():
            ratios: {str: float} = {x.algorithm: x.ratio for x in self.get_ratios_for_file(f.hash)}

            for c in compressors:
                if c.name not in ratios.keys():
                    # noinspection PyTypeChecker
                    future = client.submit(self._ratio_file, c, f)
                    ratio_tasks.append(future)
                    submitted_ratio_tasks += 1

        for future, result in dask.distributed.as_completed(ratio_tasks, with_results=True):
            self.update_ratio(
                Ratio(result[0].hash, result[1].name, result[2]))
            completed_ratio_tasks += 1
            logging.info(
                f"{completed_ratio_tasks}/{submitted_ratio_tasks} tasks complete, {completed_ratio_tasks / submitted_ratio_tasks * 100:.2f}%")

        logging.info(
            f"Calculated {completed_ratio_tasks} compression ratios in {default_timer() - ratio_start_time:.3f} seconds")
