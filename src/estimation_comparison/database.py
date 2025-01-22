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
import csv
import hashlib
import logging
import os
import pickle
import sqlite3
from pathlib import Path
from timeit import default_timer
from typing import Tuple, List

import dask.distributed

from estimation_comparison.model import InputFile, FriendlyRatio, FriendlyMetric, Compressor, Estimator, \
    Preprocessor, EstimationResult, CompressionResult


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
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS tag_types(
                tag_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                tag_name TEXT NOT NULL UNIQUE)
            """)
        self.con.commit()
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS file_tags(
                file_hash REFERENCES files(file_hash) NOT NULL, 
                tag_id REFERENCES tag_types(tag_id) NOT NULL,
                PRIMARY KEY (file_hash, tag_id)
                )
            """)
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS compression_results(
                file_hash REFERENCES files(file_hash) NOT NULL, 
                compressor_id REFERENCES compressors(compressor_id) NOT NULL, 
                size_bytes INTEGER
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
                metric REAL
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

    def update_compression_result(self, new_result: CompressionResult):
        try:
            self.con.execute(
                """INSERT INTO compression_results VALUES(:hash, (SELECT compressor_id FROM compressors WHERE name=:compressor_name), :size_bytes)""",
                new_result
            )
            self.con.commit()
        except sqlite3.Error as e:
            logging.exception(e)

    def get_all_files(self):
        return [InputFile(*row) for row in self.con.execute("SELECT file_hash, path, name FROM files").fetchall()]

    @property
    def input_file_count(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    def get_compression_results_for_file(self, file_hash: str):
        results = []
        for row in self.con.execute(
                """SELECT file_hash,
                 (SELECT name FROM compressors WHERE compression_results.compressor_id=compressors.compressor_id),
                 size_bytes
                 FROM compression_results WHERE file_hash = ?""",
                (file_hash,)).fetchall():
            results.append(CompressionResult(*row))
        return results

    def get_all_compression_results(self):
        results = []
        for row in self.con.execute(
                """SELECT 
                   (SELECT name FROM files where compression_results.file_hash=files.file_hash),
                   (SELECT name FROM compressors WHERE compression_results.compressor_id=compressors.compressor_id),
                   size_bytes
                   FROM compression_results""").fetchall():
            results.append(FriendlyRatio(*row, ))
        return results

    def get_missing_compression_results(self) -> (InputFile, str):
        results = []
        for row in self.con.execute(
                """
                SELECT combi.file_hash, combi.file_path, combi.file_name, combi.compressor_name
                FROM (
                    SELECT DISTINCT 
                        file_hash, 
                        compressor_id, 
                        files.name AS file_name,
                        files.path AS file_path,
                        compressors.name AS compressor_name
                    FROM files
                    CROSS JOIN compressors
                ) combi
                LEFT JOIN compression_results AS c ON 
                    c.file_hash = combi.file_hash AND c.compressor_id = combi.compressor_id
                WHERE c.size_bytes IS NULL"""
        ).fetchall():
            results.append((InputFile(row[0], row[1], row[2]), row[3]))
        return results

    def get_missing_estimation_results(self) -> (InputFile, str, str):
        results = []
        for row in self.con.execute(
                """
                SELECT final.file_hash, final.file_path, final.file_name, final.preprocessor_name, final.estimator_name
                FROM ((
                    SELECT DISTINCT 
                        file_hash, 
                        estimator_id,
                        preprocessor_id,
                        files.name AS file_name,
                        files.path AS file_path,
                        preprocessors.name AS preprocessor_name,
                        estimators.name AS estimator_name
                    FROM files
                    CROSS JOIN estimators
                    CROSS JOIN preprocessors
                ) AS perm
                LEFT JOIN file_estimations AS e ON 
                    (e.file_hash = perm.file_hash AND 
                    e.preprocessor_id = perm.preprocessor_id AND 
                    e.estimator_id = perm.estimator_id)) as final
                WHERE final.metric IS NULL"""
        ).fetchall():
            results.append((InputFile(row[0], row[1], row[2]), row[3], row[4]))
        return results

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

    def update_result(self, result: EstimationResult):
        try:
            self.con.execute(
                """INSERT INTO file_estimations VALUES(?, 
                                                          (SELECT preprocessor_id FROM preprocessors WHERE name=?), 
                                                          (SELECT estimator_id FROM estimators WHERE name=?),
                                                          ?)""",
                (result.input_file.hash, result.preprocessor.name, result.estimator.name,
                 result.value[0] if isinstance(result.value, list) and len(result.value) == 1 else result.value)
            )
            self.con.commit()
        except sqlite3.Error as e:
            logging.exception(e, result)

    def get_all_metric(self):
        metrics = []
        for row in self.con.execute(
                """SELECT 
                   (SELECT name FROM files where file_estimations.file_hash=files.file_hash),
                   (SELECT name FROM preprocessors where file_estimations.preprocessor_id=preprocessors.preprocessor_id),
                   (SELECT name FROM estimators WHERE file_estimations.estimator_id=estimators.estimator_id),
                   metric
                   FROM file_estimations""").fetchall():
            metrics.append(FriendlyMetric(file_name=row[0], preprocessor=row[1], estimator=row[2],
                                          metric=row[3] if not isinstance(row[3], bytes) else pickle.loads(row[3])))
        return metrics

    def get_dataframe(self):
        cursor = self.con.cursor()
        cursor.execute("""SELECT (SELECT name FROM files WHERE compression_results.file_hash = files.file_hash) as filename,
                                 (SELECT name FROM preprocessors WHERE file_estimations.preprocessor_id = preprocessors.preprocessor_id) as preprocessor,
                                 (SELECT name FROM estimators WHERE file_estimations.estimator_id = estimators.estimator_id) as estimator,
                                 (SELECT name FROM compressors WHERE compression_results.compressor_id = compressors.compressor_id) as compressor,
                                 compression_results.size_bytes as size_bytes,
                                 file_estimations.metric as metric
                          FROM compression_results
                                 LEFT OUTER JOIN file_estimations on compression_results.file_hash = file_estimations.file_hash
                          WHERE metric IS NOT NULL""")
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

    def update_tags(self, tags_csv: Path) -> List[str]:
        try:
            with open(tags_csv, "r") as f:
                reader = csv.DictReader(f, restkey="Tags")
                unique_tags = set()
                file_tags: List[Tuple[str, str]] = []
                for row in reader:
                    tags: List[str] = (row["Keywords"] + (row["Tags"] if "Tags" in row.keys() else "")
                            ).replace(";", "").split(" ")
                    unique_tags.update(map(lambda x: (x.lower(),), tags))

                    for tag in tags:
                        file_tags.append(("RAISE/" + row["File"], tag.lower()))

                self.con.executemany("INSERT OR IGNORE INTO tag_types(tag_name) VALUES(?)", unique_tags)
                self.con.commit()

                self.con.executemany("INSERT OR IGNORE INTO file_tags VALUES((SELECT file_hash FROM files WHERE ? = files.name), ?)", file_tags)
                self.con.commit()

        except:
            logging.exception(f"Error reading {tags_csv}")
            return []

    @staticmethod
    def _compress_file(compressor: Compressor, f: InputFile) -> Tuple[InputFile, Compressor, float]:
        try:
            with open(f.path, "rb") as fd:
                return f, compressor, compressor.instance.run(fd.read())
        except ValueError as e:
            logging.warning(e)

    def update_compression_results(self, client: dask.distributed.Client, compressors: List[Compressor]):
        ratio_start_time = default_timer()
        compression_tasks = []
        submitted_compression_tasks = 0
        completed_compression_tasks = 0

        for job in self.get_missing_compression_results():
            future = client.submit(self._compress_file,
                                   next(filter(lambda x: x.name == job[1], compressors)), job[0])
            compression_tasks.append(future)
            submitted_compression_tasks += 1

        for future, result in dask.distributed.as_completed(compression_tasks, with_results=True):
            self.update_compression_result(
                CompressionResult(result[0].hash, result[1].name, result[2]))
            completed_compression_tasks += 1
            logging.info(
                f"{completed_compression_tasks}/{submitted_compression_tasks} tasks complete, {completed_compression_tasks / submitted_compression_tasks * 100:.2f}%")

        logging.info(
            f"Calculated {completed_compression_tasks} new compression ratios in {default_timer() - ratio_start_time:.3f} seconds")

    # def check_estimation_done(self, preprocessor: str, estimator: str, compressor: Compressor) -> bool:
    #     return self.con.execute("""SELECT COUNT(1) FROM file_estimations WHERE
    #                                file_estimations.preprocessor_id=(SELECT preprocessor_id from preprocessors WHERE name=?) AND
    #     """, ()).fetchone() > 0
