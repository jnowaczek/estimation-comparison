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


class BenchmarkDatabase:
    def __init__(self, db_path: Path):
        self.con = sqlite3.connect(db_path)
        self.cur = self.con.cursor()

        if not db_path.exists():
            logging.warning(f"Benchmark database did not exist, creating...")
            self._create_tables()

    def _create_tables(self):
        self.cur.execute(
            "CREATE TABLE files(file_hash TEXT PRIMARY KEY NOT NULL, path TEXT NOT NULL, name TEXT NOT NULL)")
        self.cur.execute("CREATE TABLE tag_types(tag_id INT PRIMARY KEY NOT NULL, tag_name TEXT NOT NULL)")
        self.cur.execute(
            "CREATE TABLE file_tags(file_hash REFERENCES files(file_hash), tag_id REFERENCES tag_types(tag_id))")
        self.con.commit()

    def update_file(self, file: InputFile):
        try:
            self.cur.execute(
                "INSERT INTO files VALUES(:hash, :path, :name)"
                "   ON CONFLICT(file_hash) DO UPDATE SET "
                "       path = :path, "
                "       name = :name",
                file)
            self.con.commit()
        except sqlite3.Error as e:
            logging.exception(e)
