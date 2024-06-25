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
import argparse
import concurrent.futures
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import functools
from dataclasses import dataclass
from datetime import datetime
from timeit import default_timer
from pathlib import Path
from typing import List, Dict

from estimation_comparison.data_collection.compressor.gzip import GzipCompressor
from estimation_comparison.data_collection.compressor.lzma import LzmaCompressor
from estimation_comparison.data_collection.estimator import Autocorrelation, ByteCount, Entropy


@dataclass
class _InputFile:
    path: Path
    name: str


class Benchmark:
    def __init__(self, output_dir: str, workers: int | None, parallel=False):
        self.results: Dict = {}
        self.output_dir = output_dir
        self.file_list: List[_InputFile] = []
        self._estimators = {
            "autocorrelation_1k": Autocorrelation({"block_size": 1024}),
            "bytecount_1k": ByteCount({"block_size": 1024}),
            "bytecount_file": ByteCount({"block_size": None}),
            "entropy_bits": Entropy({"base": 2}),
        }
        self._compressors = {
            "gzip_max": GzipCompressor({"level": 9}),
            "lzma": LzmaCompressor({}),
        }
        self.algorithms = self._estimators | self._compressors

        if parallel:
            self.process_pool = ProcessPoolExecutor(max_workers=workers)
        else:
            self.process_pool = None

    def build_file_list(self, locations: List[str]):
        for s in locations:
            path = Path(s)
            logging.debug(f"Entering directory '{path}'")
            for file in filter(lambda f: f.is_file(), path.glob("**/*")):
                self.file_list.append(_InputFile(file, os.path.relpath(file, path)))
                logging.debug(f"Adding file '{file}'")
        logging.info(
            f"Collected {len(self.file_list)} input file{"s" if len(self.file_list) > 1 else ""} from"
            f" {len(locations)} directory{"s" if len(locations) > 1 else ""}")

        for file in self.file_list:
            self.results[file.name] = {}

    def run(self):
        start_time = default_timer()
        num_tasks = len(self.file_list) * len(self.algorithms.values())
        completed_tasks = 0

        tasks = []
        for instance_name, instance in self.algorithms.items():
            for file in self.file_list:
                try:
                    data = self._read_cached(file.path)
                    if self.process_pool is not None:
                        future = self.process_pool.submit(instance.run, data)
                        future.context = (instance_name, instance, file)
                        tasks.append(future)
                    else:
                        result = instance.run(data)
                        self.results[file.name] |= {f"{instance_name} Parameters": instance.parameters,
                                                    f"{instance_name}": result}
                        completed_tasks += 1
                        logging.info(
                            f"{completed_tasks}/{num_tasks} tasks complete, {completed_tasks / num_tasks * 100:.2f}%")

                except Exception as e:
                    logging.exception(f"Input file '{file.name}' raised exception\n\t{e})")

        if self.process_pool is not None:
            for future in concurrent.futures.as_completed(tasks):
                completed_tasks += 1
                logging.info(f"{completed_tasks}/{num_tasks} tasks complete, {completed_tasks / num_tasks * 100:.2f}%")
                self.results[future.context[2].name] |= {
                    f"{future.context[0]} Parameters": future.context[1].parameters,
                    f"{future.context[0]}": future.result()}
            self.process_pool.shutdown()

        logging.info(f"Benchmark completed in {default_timer() - start_time:.3f} seconds")

        output_file = f"{self.output_dir}/benchmark_{datetime.now().isoformat()}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(self.results, f)
            logging.info(f"Results written to '{output_file}'")

    @functools.cache
    def _read_cached(self, path):
        with open(path, "rb") as f:
            return f.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", action="append")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    parser.add_argument("-p", "--parallel", dest="parallel", action="store_true")
    parser.add_argument("-w", "--workers", type=int, dest="workers", default=None)
    parser.add_argument("-o", "--output_dir", type=str, dest="output_dir", default=".")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    benchmark = Benchmark(args.output_dir, workers=args.workers, parallel=args.parallel)
    benchmark.build_file_list(args.dir)

    benchmark.run()
