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
import argparse
import csv
import logging
import random
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from hashlib import file_digest
from io import BytesIO
from pathlib import Path

import requests
from wand.image import Image

# SHA256 hashes of the RAISE dataset CSV: http://loki.disi.unitn.it/RAISE/download.html
RAISE_1K_SHA256 = "b388d84a73b17f3c1790a55e28777c6b9bed2806b2a86b9fc05ed28b863cc8c7"
RAISE_ALL_SHA256 = "2d78a40847564a4d2e810377b8b208c7207c25d11e1b7b2b10c6de5395c760d3"
RAISE_HASHES = [RAISE_1K_SHA256, RAISE_ALL_SHA256]


class RaiseDownloader:
    def __init__(self, csv_path: Path, skip_hash_check: bool, output_dir: Path):
        self.path = csv_path
        self.force_download = skip_hash_check
        self.output_dir = output_dir
        self.filelist = None

        if not self.output_dir.exists():
            self.output_dir.mkdir()

    def run(self):
        try:
            if not self.force_download:
                with open(self.path, "rb") as f:
                    if not file_digest(f, "sha256").hexdigest() in RAISE_HASHES:
                        logging.error(f"CSV hash does not match known dataset hash, --skip-hash-check to override")
                        exit(1)

            with open(self.path, "r", newline="") as f:
                reader = csv.DictReader(f)
                self.filelist = list(reader)
                # Try to be nice and not download images in the same order every time
                random.shuffle(self.filelist)
        except OSError as e:
            logging.exception(f"Unable to read file list: {e}")
            exit(1)

        with ProcessPoolExecutor() as executor:
            completed_tasks = 0
            tasks = []

            for file in self.filelist:
                tasks.append(executor.submit(self._get_and_magick_file, file["TIFF"], self.output_dir / file["File"]))

            for _ in futures.as_completed(tasks):
                completed_tasks += 1
                logging.info(
                    f"{completed_tasks}/{len(self.filelist)} downloaded and cleaned, "
                    f"{completed_tasks / len(self.filelist) * 100:.2f}%")
            executor.shutdown()

    @staticmethod
    def _get_and_magick_file(url: str, dest_path: Path):
        r = requests.get(url)
        with Image(blob=r.content, format="tiff") as img:
            # Remove metadata to prevent it from skewing compression ratio
            img.strip()
            # Remove all but the first TIFF layer
            for x in range(1, len(img.sequence)):
                del img.sequence[x]
            with open(dest_path, "wb") as f:
                img.save(file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    parser.add_argument("-s", "--skip-hash-check", dest="skip_hash_check", action="store_true", default=False)
    parser.add_argument(type=Path, dest="csv_path")
    parser.add_argument(type=Path, dest="output_dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    d = RaiseDownloader(args.csv_path, args.skip_hash_check, args.output_dir)

    d.run()


if __name__ == "__main__":
    main()
