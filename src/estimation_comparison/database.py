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
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.#
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
from typing import Tuple, List, Optional

import dask.distributed

from estimation_comparison.model import InputFile, Compressor, Estimator, \
    Preprocessor, EstimationResult, CompressionResult, FileSummaryFunc, BlockSummaryFunc, EstimationTask, \
    CompressionTask


class BenchmarkDatabase:
    def __init__(self, db_path: Path):
        self.con = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS files
            (
                file_hash  TEXT PRIMARY KEY NOT NULL,
                path       TEXT             NOT NULL,
                name       TEXT             NOT NULL,
                size_bytes INTEGER          NOT NULL
            )
            """)
        self.con.commit()
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS compressors
            (
                compressor_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                name          TEXT                              NOT NULL UNIQUE,
                parameters    BLOB
            )
            """)
        self.con.commit()
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS tag_types
            (
                tag_id   INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                tag_name TEXT                              NOT NULL UNIQUE
            )
            """)
        self.con.commit()
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS file_tags
            (
                file_hash REFERENCES files (file_hash) NOT NULL,
                tag_id REFERENCES tag_types (tag_id)   NOT NULL,
                PRIMARY KEY (file_hash, tag_id)
            )
            """)
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS compression_results
            (
                file_hash REFERENCES files (file_hash)               NOT NULL,
                compressor_id REFERENCES compressors (compressor_id) NOT NULL,
                size_bytes INTEGER
            )
            """)
        self.con.commit()
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS estimators
            (
                estimator_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                name         TEXT                              NOT NULL UNIQUE,
                parameters   BLOB,
                summarize_block BOOLEAN NOT NULL,
                summarize_file BOOLEAN NOT NULL
            )
            """)
        self.con.commit()
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS file_estimations
            (
                file_hash REFERENCES files (file_hash)                     NOT NULL,
                preprocessor_id REFERENCES preprocessors (preprocessor_id) NOT NULL,
                estimator_id REFERENCES estimators (estimator_id)          NOT NULL,
                block_summary_func_id REFERENCES block_summary_funcs (block_summary_id),
                file_summary_func_id REFERENCES file_summary_funcs (file_summary_id),
                metric REAL                                                NOT NULL,
                UNIQUE (file_hash, preprocessor_id, estimator_id, block_summary_func_id, file_summary_func_id) ON CONFLICT REPLACE
            )
            """)
        self.con.commit()
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS preprocessors
            (
                preprocessor_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                name            TEXT                              NOT NULL UNIQUE,
                parameters      BLOB
            )
            """)
        self.con.commit()
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS block_summary_funcs
            (
                block_summary_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                name             TEXT                              NOT NULL UNIQUE,
                parameters       BLOB
            )
            """
        )
        self.con.commit()
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS file_summary_funcs
            (
                file_summary_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                name            TEXT                              NOT NULL UNIQUE,
                parameters      BLOB
            )
            """
        )
        self.con.commit()

    def update_estimators(self, estimators: List[Estimator]):
        try:
            estimator_list = []
            for e in estimators:
                estimator_list.append(
                    {"name": e.name, "parameters": pickle.dumps(e.instance.traits(), protocol=pickle.HIGHEST_PROTOCOL),
                     "summarize_block": e.summarize_block, "summarize_file": e.summarize_file})

            self.con.executemany(
                """
                INSERT OR IGNORE INTO estimators(name, parameters, summarize_block, summarize_file)
                VALUES (:name, :parameters, :summarize_block, :summarize_file)
                """, estimator_list)
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

    def update_block_summary_funcs(self, block_funcs: List[BlockSummaryFunc]):
        try:
            block_func_list = []
            for f in block_funcs:
                block_func_list.append(
                    {"name": f.name, "parameters": pickle.dumps(f.parameters, protocol=pickle.HIGHEST_PROTOCOL)})

            self.con.executemany(
                "INSERT OR IGNORE INTO block_summary_funcs(name, parameters) VALUES(:name, :parameters)",
                block_func_list)
            self.con.commit()
        except sqlite3.Error as e:
            logging.exception(e)

    def update_file_summary_funcs(self, file_funcs: List[FileSummaryFunc]):
        try:
            file_func_list = []
            for f in file_funcs:
                file_func_list.append(
                    {"name": f.name, "parameters": pickle.dumps(f.parameters, protocol=pickle.HIGHEST_PROTOCOL)})

            self.con.executemany(
                "INSERT OR IGNORE INTO file_summary_funcs(name, parameters) VALUES(:name, :parameters)",
                file_func_list)
            self.con.commit()
        except sqlite3.Error as e:
            logging.exception(e)

    def update_file(self, file: InputFile):
        self.con.execute(
            """
                INSERT OR ABORT INTO files
                VALUES (:hash, :path, :name, :size_bytes)
                """, (file.hash, file.path, file.name, file.size_bytes))
        self.con.commit()

    def update_files(self, client: dask.distributed.Client, locations):
        hash_tasks = []
        duplicate_files = 0

        # noinspection SqlWithoutWhere
        self.con.execute("DELETE FROM files")
        self.con.commit()

        # Glob 'em, hash 'em, and INSERT 'em
        for s in locations:
            path = Path(s)
            logging.debug(f"Entering directory '{path}'")
            for file in filter(lambda f: f.is_file(), path.glob("**/*")):
                future = client.submit(self._hash_file, file)
                future.context = (str(file), os.path.relpath(file, path), file.stat().st_size)
                hash_tasks.append(future)

        for future, result in dask.distributed.as_completed(hash_tasks, with_results=True):
            try:
                self.update_file(InputFile(result, future.context[0], future.context[1], future.context[2]))
            except sqlite3.IntegrityError as e:
                logging.debug(f"Ignoring file '{future.context[1]}': {e}")
                duplicate_files += 1

        if duplicate_files:
            logging.warning(
                f"Ignored {duplicate_files} files with hash collisions. Debug mode (-v) can provide additional details")
        logging.info(f"Finished hashing {len(hash_tasks)} files")

    def update_tags(self, tags_csv: Path):
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

                self.con.executemany(
                    """
                    INSERT OR IGNORE INTO file_tags
                    VALUES ((SELECT file_hash FROM files WHERE ? = files.name),
                            (SELECT tag_id from tag_types WHERE ? = tag_types.tag_name))
                    """, file_tags)
                self.con.commit()
        except OSError:
            logging.exception(f"Error reading {tags_csv}")

    def update_compression_result(self, new_result: CompressionResult):
        try:
            self.con.execute(
                """INSERT INTO compression_results
                       VALUES (:hash, (SELECT compressor_id FROM compressors WHERE name = :compressor_name), :size_bytes)
                       """, (new_result.input_file.hash, new_result.compressor.name, new_result.input_file.size_bytes))
            self.con.commit()
        except sqlite3.Error as e:
            logging.exception(e)

    def update_compression_results(self, client: dask.distributed.Client, compressors: List[Compressor]):
        ratio_start_time = default_timer()
        compression_tasks = []
        submitted_compression_tasks = 0
        completed_compression_tasks = 0

        for job in self.get_missing_compression_results():
            future = client.submit(self._compress_file,
                                   compressor=next(filter(lambda x: x.name == job.compressor_name, compressors)),
                                   f=job.input_file)
            compression_tasks.append(future)
            submitted_compression_tasks += 1

        for future, result in dask.distributed.as_completed(compression_tasks, with_results=True):
            if isinstance(result, CompressionResult):
                self.update_compression_result(
                    CompressionResult(input_file=result.input_file, compressor=result.compressor,
                                      compressed_size_bytes=result.compressed_size_bytes))
                completed_compression_tasks += 1
                logging.info(
                    f"{completed_compression_tasks}/{submitted_compression_tasks} tasks complete, {completed_compression_tasks / submitted_compression_tasks * 100:.2f}%")

        logging.info(
            f"Calculated {completed_compression_tasks} new compression ratios in {default_timer() - ratio_start_time:.3f} seconds")

    def get_all_files(self):
        return [InputFile(*row) for row in
                self.con.execute("SELECT file_hash, path, name, size_bytes FROM files").fetchall()]

    @property
    def input_file_count(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    def get_compression_results_for_file(self, file_hash: str) -> List[CompressionResult]:
        results = []
        for row in self.con.execute(
                """
                    SELECT file_hash,
                           (SELECT name FROM compressors WHERE compression_results.compressor_id = compressors.compressor_id),
                           size_bytes
                    FROM compression_results
                    WHERE file_hash = ?
                    """,
                (file_hash,)).fetchall():
            results.append(CompressionResult(*row))
        return results

    def get_missing_compression_results(self) -> List[CompressionTask]:
        results: [CompressionTask] = []
        for row in self.con.execute(
                """
                SELECT combi.file_hash, combi.file_name, combi.file_path, combi.uncompressed_size_bytes, combi.compressor_name
                FROM (SELECT DISTINCT file_hash,
                                      files.path       AS file_path,
                                      files.name       AS file_name,
                                      files.size_bytes AS uncompressed_size_bytes,
                                      compressor_id,
                                      compressors.name AS compressor_name
                      FROM files
                               CROSS JOIN compressors) combi
                         LEFT JOIN compression_results AS c ON
                    c.file_hash = combi.file_hash AND c.compressor_id = combi.compressor_id
                WHERE c.size_bytes IS NULL
                """).fetchall():
            results.append(
                CompressionTask(input_file=InputFile(hash=row[0], name=row[1], path=row[2], size_bytes=row[3]),
                                compressor_name=row[4]))
        return results

    def get_missing_estimation_results(self) -> List[EstimationTask]:
        results = []
        # Phew, all hail the mighty CTE
        for row in self.con.execute(
                """
                WITH estimators_without_summary_func as (SELECT estimator_id, name AS est_name
                                                         FROM estimators
                                                         WHERE summarize_block = FALSE
                                                           AND summarize_file = FALSE),
                     estimators_with_file_summary_func AS (SELECT estimator_id, name AS est_name
                                                           FROM estimators
                                                           WHERE summarize_block = FALSE
                                                             AND summarize_file = TRUE),
                     estimators_with_block_summary_func AS (SELECT estimator_id, name AS est_name
                                                            FROM estimators
                                                            WHERE summarize_block = TRUE
                                                              AND summarize_file = TRUE),
                     file_summary_permutations AS (SELECT estimator_id, est_name, fsf.name AS fsf_name
                                                   FROM estimators_with_file_summary_func
                                                            CROSS JOIN file_summary_funcs fsf),
                     block_summary_permutations AS (SELECT estimator_id, est_name, bsf.name AS bsf_name, fsf.name AS fsf_name
                                                    FROM estimators_with_block_summary_func
                                                             CROSS JOIN block_summary_funcs bsf
                                                             CROSS JOIN file_summary_funcs fsf),
                     estimator_permutations AS (SELECT estimator_id,
                                                       est_name,
                                                       NULL AS bsf_name,
                                                       NULL AS fsf_name
                                                FROM estimators_without_summary_func
                                                UNION
                                                SELECT estimator_id, est_name, NULL AS bsf_name, fsf_name
                                                FROM file_summary_permutations
                                                UNION
                                                SELECT estimator_id, est_name, bsf_name, fsf_name
                                                FROM block_summary_permutations)
                SELECT r.fh,
                       r.file_path,
                       r.file_name,
                       r.uncompressed_size_bytes,
                       r.preprocessor_name,
                       r.estimator_name,
                       r.block_summary_func_name,
                       r.file_summary_func_name
                FROM ((SELECT DISTINCT file_hash          AS fh,
                                       ep.estimator_id,
                                       preprocessor_id,
                                       files.name         AS file_name,
                                       files.path         AS file_path,
                                       files.size_bytes   as uncompressed_size_bytes,
                                       preprocessors.name AS preprocessor_name,
                                       ep.est_name        AS estimator_name,
                                       bsf_name           AS block_summary_func_name,
                                       fsf_name           AS file_summary_func_name
                       FROM files
                                CROSS JOIN estimator_permutations ep
                                CROSS JOIN preprocessors) permutations
                    LEFT JOIN file_estimations ON
                    (file_estimations.file_hash = permutations.fh AND
                     file_estimations.preprocessor_id = permutations.preprocessor_id AND
                     file_estimations.estimator_id = permutations.estimator_id)) r
                WHERE r.metric IS NULL
                """
        ).fetchall():
            results.append(
                EstimationTask(input_file=InputFile(hash=row[0], path=row[1], name=row[2], size_bytes=row[3]),
                               preprocessor_name=row[4], estimator_name=row[5], block_summary_func_name=row[6],
                               file_summary_func_name=row[7]))
        return results

    def update_estimation_result(self, result: EstimationResult):
        try:
            self.con.execute(
                """
                INSERT INTO file_estimations
                VALUES (?,
                        (SELECT preprocessor_id FROM preprocessors WHERE name = ?),
                        (SELECT estimator_id FROM estimators WHERE name = ?),
                        (SELECT block_summary_func_id FROM block_summary_funcs WHERE name = ?),
                        (SELECT file_summary_func_id FROM file_summary_funcs WHERE name = ?),
                        ?)
                """,
                (result.input_file.hash, result.preprocessor.name, result.estimator.name,
                 result.block_summary_func.name, result.file_summary_func.name,
                 result.value[0] if isinstance(result.value, list) and len(result.value) == 1 else result.value)
            )
            self.con.commit()
        except sqlite3.Error as e:
            logging.exception(e, result)

    def get_preprocessors(self) -> List[Tuple[int, str]]:
        return self.con.execute(
            """
            SELECT preprocessor_id, name
            FROM preprocessors
            """).fetchall()

    def get_estimators(self) -> List[Tuple[int, str]]:
        return self.con.execute(
            """
            SELECT estimator_id, name
            FROM estimators
            """).fetchall()

    def get_compressors(self) -> List[Tuple[int, str]]:
        return self.con.execute(
            """
            SELECT compressor_id, name
            FROM compressors
            """).fetchall()

    def get_block_summary_funcs(self) -> List[Tuple[int, str]]:
        return self.con.execute(
            """
            SELECT block_summary_id, name
            FROM block_summary_funcs
            """).fetchall()

    def get_file_summary_funcs(self) -> List[Tuple[int, str]]:
        return self.con.execute(
            """
            SELECT file_summary_id, name
            FROM file_summary_funcs
            """).fetchall()

    def get_tags(self) -> List[Tuple[int, str]]:
        return self.con.execute("SELECT tag_id, tag_name FROM tag_types").fetchall()

    def get_combinations(self) -> List[Tuple[str, str, str]]:
        return self.con.execute(
            """
            SELECT p.name, e.name, c.name
            FROM preprocessors p
                     CROSS JOIN estimators e
                     CROSS JOIN compressors c
            """).fetchall()

    def get_all_estimations_dataframe(self):
        cursor = self.con.execute(
            """
            SELECT compression_results.file_hash,
                   preprocessor_id,
                   estimator_id,
                   compressor_id,
                   metric,
                   files.size_bytes AS initial_size,
                   compression_results.size_bytes AS final_size
            FROM compression_results
                     INNER JOIN file_estimations ON compression_results.file_hash = file_estimations.file_hash
                     INNER JOIN files ON compression_results.file_hash = files.file_hash
            WHERE metric IS NOT NULL
            """)
        return cursor.description, cursor.fetchall()

    def get_solo_tag_plot_dataframe(self, preprocessor: str, estimator: str, compressor: str, tag: str):
        cursor = self.con.execute(
            """
            WITH files_with_tag AS (SELECT file_hash
                                    FROM file_tags ft
                                             JOIN tag_types tt ON ft.tag_id = tt.tag_id
                                    GROUP BY file_hash
                                    HAVING SUM(CASE WHEN tag_name = ? THEN 1 ELSE 0 END) > 0),
                 filtered_files AS (SELECT f.*
                                    FROM files f
                                             INNER JOIN files_with_tag fwt ON fwt.file_hash = f.file_hash)
            SELECT fe.file_hash,
                   fe.metric,
                   ff.size_bytes as initial_size,
                   cr.size_bytes as final_size
            FROM file_estimations fe
                     INNER JOIN filtered_files ff ON ff.file_hash = fe.file_hash
                     INNER JOIN compression_results cr ON cr.file_hash = fe.file_hash
            WHERE metric IS NOT NULL
              AND fe.preprocessor_id = (SELECT preprocessor_id FROM preprocessors WHERE name = ?)
              AND fe.estimator_id = (SELECT estimator_id FROM estimators WHERE name = ?)
              AND cr.compressor_id = (SELECT compressor_id FROM compressors WHERE name = ?)
            ORDER BY name
            """, (tag, preprocessor, estimator, compressor))
        return cursor.description, cursor.fetchall()

    def get_solo_plot_dataframe(self, preprocessor: str, estimator: str, compressor: str):
        cursor = self.con.execute(
            """
                SELECT fe.file_hash,
                       fe.metric,
                       f.size_bytes as initial_size,
                       cr.size_bytes as final_size
                FROM file_estimations fe
                         INNER JOIN files f ON f.file_hash = fe.file_hash
                         INNER JOIN compression_results cr ON cr.file_hash = fe.file_hash
                WHERE metric IS NOT NULL
                  AND fe.preprocessor_id = (SELECT preprocessor_id FROM preprocessors WHERE name = ?)
                  AND fe.estimator_id = (SELECT estimator_id FROM estimators WHERE name = ?)
                  AND cr.compressor_id = (SELECT compressor_id FROM compressors WHERE name = ?)
                ORDER BY name
                """, (preprocessor, estimator, compressor))
        return cursor.description, cursor.fetchall()

    @staticmethod
    def _hash_file(p: Path) -> Optional[str]:
        try:
            with open(p, "rb") as f:
                # noinspection PyTypeChecker
                return hashlib.file_digest(f, hashlib.sha256).hexdigest()
        except Exception as e:
            logging.exception(e)

    @staticmethod
    def _compress_file(compressor: Compressor, f: InputFile) -> Optional[CompressionResult]:
        try:
            with open(f.path, "rb") as fd:
                return CompressionResult(input_file=f, compressor=compressor,
                                         compressed_size_bytes=compressor.instance.run(fd.read()))
        except (OSError, ValueError) as e:
            logging.exception(e)
