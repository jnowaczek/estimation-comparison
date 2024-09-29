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
import logging
import sqlite3
from pathlib import Path
from typing import NamedTuple

InputFile = NamedTuple("InputFile", [("hash", str), ("path", str), ("name", str)])
Ratio = NamedTuple("Ratio", [("algorithm", str), ("ratio", float)])


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
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS compressors(
                compressor_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, 
                name TEXT NOT NULL UNIQUE
            )""")
        # self.cur.execute("CREATE TABLE tag_types(tag_id INT PRIMARY KEY NOT NULL, tag_name TEXT NOT NULL)")
        # self.cur.execute(
        #     "CREATE TABLE file_tags(file_hash REFERENCES files(file_hash), tag_id REFERENCES tag_types(tag_id))")
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS file_ratios(
                file_hash REFERENCES files(file_hash), 
                compressor_id REFERENCES compressors(compressor_id), 
                ratio FLOAT
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

    def get_all_files(self):
        return [InputFile(*row) for row in self.con.execute("SELECT file_hash, path, name FROM files").fetchall()]

    def get_ratios_for_file(self, file_hash: str):
        return [Ratio(*row) for row in
                self.con.execute("SELECT * FROM files WHERE file_hash = ?", (file_hash,)).fetchall()]
