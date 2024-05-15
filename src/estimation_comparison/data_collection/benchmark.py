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
from pathlib import Path
from typing import List, Dict

from estimation_comparison.data_collection.estimator import Autocorrelation, ByteCount, Entropy


class Benchmark:
    def __init__(self, workers: int, parallel=False):
        self.results: Dict = {}
        self.file_list: List[Path] = []
        self.estimators = {
            "autocorrelation_1k": Autocorrelation({"block_size": 1024}),
            "bytecount_1k": ByteCount({"block_size": 1024}),
            "bytecount_file": ByteCount({"block_size": None}),
            "entropy_bits": Entropy({"base": 2})
        }

        for key in self.estimators:
            self.results[key] = {}

        if parallel:
            self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        else:
            self.process_pool = None

    def setup_algorithms(self):
        pass

    def build_file_list(self, locations: List[str]):
        for s in locations:
            path = Path(s)
            logging.debug(f"Entering directory '{path}'")
            for file in filter(lambda f: f.is_file(), path.glob("**/*")):
                self.file_list.append(file)
                logging.debug(f"Adding file '{file}'")
        logging.info(
            f"Collected {len(self.file_list)} input file{"s" if len(self.file_list) > 1 else ""} from"
            f" {len(locations)} directory{"s" if len(locations) > 1 else ""}")

    def run(self):
        for instance_name, instance in self.estimators.items():
            for file in self.file_list:
                with file.open("rb") as f:
                    try:
                        if self.process_pool is not None:
                            result = self.process_pool.submit(instance.estimate, f.read()).result()
                            print(result)
                        else:
                            result = instance.estimate(f.read())
                            self.results[instance_name][file] = result

                    except Exception as e:
                        logging.exception(f"Input file '{file.name}' raised exception\n\t{e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", action="append")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    parser.add_argument("-p", "--parallel", dest="parallel", action="store_true")
    parser.add_argument("-w", "--workers", type=int, dest="workers", default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    benchmark = Benchmark(workers=args.workers, parallel=args.parallel)
    benchmark.build_file_list(args.dir)

    benchmark.run()
