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
import functools
import logging
from pathlib import Path
from timeit import default_timer
from typing import List

import numpy as np
from dask.distributed import Client, as_completed
from imagecodecs import tiff_check, tiff_decode

from estimation_comparison.data_collection.compressor.general import *
from estimation_comparison.data_collection.compressor.image import *
from estimation_comparison.data_collection.estimator import *
from estimation_comparison.data_collection.preprocessor import NoopSampler, PatchSampler
from estimation_comparison.data_collection.summary_stats import max_outside_middle_notch
from estimation_comparison.database import BenchmarkDatabase
from estimation_comparison.model import Compressor, Estimator, Preprocessor, InputFile, IntermediateResult, \
    Result, LoadedData


class Benchmark:
    def __init__(self, input_dir: List[str], output_dir: str):
        self._init_time = default_timer()
        self.data_locations = input_dir
        self.output_dir = output_dir
        self.database = BenchmarkDatabase(Path(self.output_dir) / "benchmark.sqlite")

        self._preprocessors: List[Preprocessor] = [
            Preprocessor(name="noop", instance=NoopSampler()),
            Preprocessor(name="patch_random", instance=PatchSampler())
        ]

        self._estimators: List[Estimator] = [
            Estimator(
                name="autocorrelation_1k_64_notch_mean",
                instance=Autocorrelation(
                    block_summary_fn=functools.partial(max_outside_middle_notch, notch_width=64),
                    file_summary_fn=np.mean
                )
            ),
            Estimator(
                name="autocorrelation_972_64_notch_mean",
                instance=Autocorrelation(
                    block_size=972,
                    block_summary_fn=functools.partial(max_outside_middle_notch, notch_width=64),
                    file_summary_fn=np.mean
                )
            ),
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
            Estimator(name="entropy_bits", instance=Entropy()),
        ]

        self._compressors: List[Compressor] = [
            Compressor(name="gzip_9", instance=GzipCompressor(level=9)),
            Compressor(name="jxl_lossless", instance=JpegXlCompressor(lossless=True)),
            Compressor(name="jpeg", instance=JpegCompressor()),
            Compressor(name="jpeg2k", instance=Jpeg2kCompressor(lossless=False, level=90)),
            Compressor(name="jpeg2k_lossless", instance=Jpeg2kCompressor(lossless=True)),
            Compressor(name="lzma", instance=LzmaCompressor()),
            Compressor(name="png", instance=PngCompressor()),
            Compressor(name="bzip2_9", instance=Bzip2Compressor(level=9)),
            Compressor(name="zstd", instance=ZstandardCompressor()),
        ]

        self.client = Client()

    def update_database(self):
        logging.info("Updating benchmark database compressor list")
        self.database.update_compressors(self._compressors)
        logging.info("Updating benchmark database estimator list")
        self.database.update_estimators(self._estimators)
        logging.info("Updating benchmark database preprocessor list")
        self.database.update_preprocessors(self._preprocessors)
        logging.info("Updating benchmark database file hash list")
        self.database.update_files(self.client, self.data_locations)
        logging.info("Updating benchmark database compression ratios")
        self.database.update_ratios(self.client, self._compressors)

    @staticmethod
    def _load_file(file: InputFile) -> LoadedData:
        try:
            with open(file.path, "rb") as f:
                data = f.read()
                if tiff_check(data):
                    return LoadedData(data=tiff_decode(data), completed_stages=["tiff_decode"], input_file=file)
                else:
                    return LoadedData(data=np.frombuffer(data), completed_stages=["load_bytes"],
                                      input_file=file)
        except OSError as e:
            logging.exception(f"Error reading {file}: {e}")

    @staticmethod
    def _preprocess_file(preprocessor: Preprocessor, data: LoadedData) -> IntermediateResult:
        return IntermediateResult(preprocessor.instance.run(data.data),
                                  completed_stages=data.completed_stages + [preprocessor.name],
                                  input_file=data.input_file, preprocessor=preprocessor)

    @staticmethod
    def _run_estimator(estimator: Estimator, ir: IntermediateResult) -> Result:
        try:
            return Result.from_intermediate_result(ir, estimator.instance.run(ir.data), estimator)
        except Exception as e:
            logging.exception(f"Error estimating {ir.input_file}: {e}")

    def run(self):
        start_time = default_timer()
        num_tasks = self.database.input_file_count * len(self._estimators)
        completed_tasks = 0

        input_files = self.database.get_all_files()
        loaded_files = self.client.map(self._load_file, input_files, priority=0)

        preprocessed = []
        for p in self._preprocessors:
            for file in loaded_files:
                preprocessed.append(self.client.submit(functools.partial(self._preprocess_file, p), file, priority=5))
        # preprocessed = self.client.map(self._preprocess_file, self._preprocessors, loaded_files)
        del loaded_files

        results = []
        for e in self._estimators:
            for future in preprocessed:
                try:
                    # noinspection PyTypeChecker
                    results.append(
                        self.client.submit(functools.partial(self._run_estimator, e), future, priority=10))
                except Exception as ex:
                    logging.exception(f"Running estimator raised exception\n\t{ex})")
        del preprocessed

        for future, result in as_completed(results, with_results=True):
            completed_tasks += 1
            logging.info(
                f"{completed_tasks}/{num_tasks} estimation tasks complete, {completed_tasks / len(results) * 100:.2f}%")
            try:
                self.database.update_result(result)
                # self.database.update_metric(
                #     Metric(result.input_file.hash, result.completed_stages[-1],
                #            pickle.dumps(result.value, pickle.HIGHEST_PROTOCOL)))
                # self.results[future.context["file"].name] |= {
                #     f"{future.context["estimator_name"]} Parameters": future.context["estimator_instance"].parameters,
                #     f"{future.context["estimator_name"]}": future.result()}
            except Exception as e:
                logging.exception(f"Input file '{result.input_file.name}' raised exception\n\t{e}")

        del results

        logging.info(f"Estimation completed in {default_timer() - start_time:.3f} seconds")
        logging.info(f"Benchmark completed in {default_timer() - self._init_time:.3f} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", action="append")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    parser.add_argument("-l", "--limit-files", type=int, dest="file_limit", default=0)
    parser.add_argument("-o", "--output_dir", type=str, dest="output_dir", default="./benchmarks")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    benchmark = Benchmark(args.dir, args.output_dir)
    benchmark.update_database()

    benchmark.run()


if __name__ == "__main__":
    main()
