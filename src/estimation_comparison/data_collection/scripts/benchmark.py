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
import concurrent.futures
import hashlib
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import functools
from timeit import default_timer
from pathlib import Path
from typing import List

import numpy as np

from estimation_comparison.data_collection.compressor.general import *
from estimation_comparison.data_collection.compressor.general.bzip2 import Bzip2Compressor
from estimation_comparison.data_collection.compressor.image import *
from estimation_comparison.database import BenchmarkDatabase, InputFile, Ratio, Metric
from estimation_comparison.data_collection.estimator import *
from estimation_comparison.data_collection.summary_stats import max_outside_middle_notch, proportion_below_cutoff, \
    max_below_cutoff


class Benchmark:
    def __init__(self, output_dir: str, workers: int | None):
        self._init_time = default_timer()
        self.output_dir = output_dir
        self.database = BenchmarkDatabase(Path(self.output_dir) / "benchmark.sqlite")
        self._estimators = {
            "autocorrelation_1k_64_notch_mean": Autocorrelation(
                block_summary_fn=functools.partial(max_outside_middle_notch, notch_width=64),
                file_summary_fn=np.mean),
            # "autocorrelation_1k_64_notch_max": Autocorrelation(
            #     block_summary_fn=functools.partial(max_outside_middle_notch, notch_width=64),
            #     file_summary_fn=np.max),
            # "autocorrelation_1k_768_cutoff_mean": Autocorrelation(
            #     block_summary_fn=functools.partial(proportion_below_cutoff, cutoff=768),
            #     file_summary_fn=np.mean),
            # "autocorrelation_1k_896_cutoff_mean": Autocorrelation(
            #     block_summary_fn=functools.partial(proportion_below_cutoff, cutoff=896),
            #     file_summary_fn=np.mean),
            # "autocorrelation_1k_768_cutoff_max": Autocorrelation(
            #     block_summary_fn=functools.partial(proportion_below_cutoff, cutoff=768),
            #     file_summary_fn=np.max),
            # "autocorrelation_1k_896_cutoff_max": Autocorrelation(
            #     block_summary_fn=functools.partial(max_below_cutoff, cutoff=896),
            #     file_summary_fn=np.max),
            # "autocorrelation_1k_768_cutoff_max_mean": Autocorrelation(
            #     block_summary_fn=functools.partial(max_below_cutoff, cutoff=768),
            #     file_summary_fn=np.mean),
            # "autocorrelation_1k_896_cutoff_max_mean": Autocorrelation(
            #     block_summary_fn=functools.partial(max_below_cutoff, cutoff=896),
            #     file_summary_fn=np.mean),
            # "autocorrelation_1k_768_cutoff_max_max": Autocorrelation(
            #     block_summary_fn=functools.partial(max_below_cutoff, cutoff=768),
            #     file_summary_fn=np.max),
            # "autocorrelation_1k_896_cutoff_max_max": Autocorrelation(
            #     block_summary_fn=functools.partial(max_below_cutoff, cutoff=896),
            #     file_summary_fn=np.max),
            # "bytecount_file": ByteCount(),
            # "entropy_bits": Entropy(),
        }
        self._compressors = {
            "gzip_9": GzipCompressor(level=9),
            "jxl_lossless": JpegXlCompressor(lossless=True),
            "jpeg": JpegCompressor(),
            "jpeg2k": Jpeg2kCompressor(lossless=False, level=90),
            "jpeg2k_lossless": Jpeg2kCompressor(lossless=True),
            "lzma": LzmaCompressor(),
            "png": PngCompressor(),
            "bzip2_9": Bzip2Compressor(level=9),
        }

        self.process_pool = ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1)

    @staticmethod
    def _hash_file(p: Path) -> str:
        with open(p, "rb") as f:
            return hashlib.file_digest(f, hashlib.sha256).hexdigest()

    def update_database(self, locations: List[str]):
        logging.info("Updating benchmark database compressor list")
        self.database.update_compressors(self._compressors.keys())
        logging.info("Updating benchmark database estimator list")
        self.database.update_estimators(self._estimators)

        logging.info("Updating benchmark database file list")
        hash_tasks = []

        # Glob 'em, hash 'em, and INSERT 'em
        for s in locations:
            path = Path(s)
            logging.debug(f"Entering directory '{path}'")
            for file in filter(lambda f: f.is_file(), path.glob("**/*")):
                future = self.process_pool.submit(self._hash_file, file)
                future.context = (str(file), os.path.relpath(file, path))
                hash_tasks.append(future)

        for future in concurrent.futures.as_completed(hash_tasks):
            self.database.update_file(
                InputFile(future.result(), future.context[0], future.context[1]))

    @staticmethod
    def _ratio_file(algorithm, p: Path) -> str:
        try:
            with open(p, "rb") as f:
                return algorithm.run(f.read())
        except ValueError as e:
            logging.warning(e)

    def update_ratio_database(self) -> int:
        ratio_tasks = []
        submitted_ratio_tasks = 0
        completed_ratio_tasks = 0

        for f in self.database.get_all_files():
            ratios: {str: float} = {x.algorithm: x.ratio for x in self.database.get_ratios_for_file(f.hash)}

            for name, instance in self._compressors.items():
                if name not in ratios.keys():
                    future = self.process_pool.submit(self._ratio_file, instance, f.path)
                    future.context = {"hash": f.hash, "compressor_name": name}
                    ratio_tasks.append(future)
                    submitted_ratio_tasks += 1

        for future in concurrent.futures.as_completed(ratio_tasks):
            self.database.update_ratio(
                Ratio(future.context["hash"], future.context["compressor_name"], future.result()))
            completed_ratio_tasks += 1
            logging.info(
                f"{completed_ratio_tasks}/{submitted_ratio_tasks} tasks complete, {completed_ratio_tasks / submitted_ratio_tasks * 100:.2f}%")

        return len(ratio_tasks)

    @staticmethod
    def _run_estimator(instance, file: InputFile):
        try:
            with open(file.path, "rb") as f:
                return instance.run(f.read())
        except OSError as e:
            logging.exception(f"Error reading data file '{file.path}': {e}")

    def run(self):
        ratio_start_time = default_timer()
        updates = self.update_ratio_database()
        logging.info(f"Calculated {updates} compression ratios in {default_timer() - ratio_start_time:.3f} seconds")

        start_time = default_timer()
        num_tasks = self.database.input_file_count * len(self._estimators.values())
        completed_tasks = 0

        tasks = []
        for file in self.database.get_all_files():
            for instance_name, instance in self._estimators.items():
                try:
                    future = self.process_pool.submit(self._run_estimator, instance, file)
                    future.context = {"estimator_name": instance_name, "estimator_instance": instance, "file": file}
                    tasks.append(future)
                except Exception as e:
                    logging.exception(f"Input file '{file.name}' raised exception\n\t{e})")

        for future in concurrent.futures.as_completed(tasks):
            completed_tasks += 1
            logging.info(
                f"{completed_tasks}/{num_tasks} estimation tasks complete, {completed_tasks / num_tasks * 100:.2f}%")
            try:
                self.database.update_metric(
                    Metric(future.context["file"].hash, future.context["estimator_name"],
                           pickle.dumps(future.result(), pickle.HIGHEST_PROTOCOL)))
                # self.results[future.context["file"].name] |= {
                #     f"{future.context["estimator_name"]} Parameters": future.context["estimator_instance"].parameters,
                #     f"{future.context["estimator_name"]}": future.result()}
            except Exception as e:
                logging.exception(f"Input file '{future.context["file"].name}' raised exception\n\t{e}")
            finally:
                del future
        self.process_pool.shutdown()

        logging.info(f"Estimation completed in {default_timer() - start_time:.3f} seconds")
        logging.info(f"Benchmark completed in {default_timer() - self._init_time:.3f} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", action="append")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    parser.add_argument("-w", "--workers", type=int, dest="workers", default=None)
    parser.add_argument("-l", "--limit-files", type=int, dest="file_limit", default=0)
    parser.add_argument("-o", "--output_dir", type=str, dest="output_dir", default="./benchmarks")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    benchmark = Benchmark(args.output_dir, workers=args.workers)
    benchmark.update_database(args.dir)

    benchmark.run()


if __name__ == "__main__":
    main()
