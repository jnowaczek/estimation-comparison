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
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
import argparse
import functools
import itertools
import logging
import pathlib
from pathlib import Path
from timeit import default_timer
from typing import List, Optional

import dask.array as da
import numpy as np
from dask.distributed import Client, as_completed
from imagecodecs import tiff_check, tiff_decode

from estimation_comparison.data_collection.compressor.general import *
from estimation_comparison.data_collection.compressor.image import *
from estimation_comparison.data_collection.estimator import *
from estimation_comparison.data_collection.preprocessor import FlattenSampler, PatchSampler
from estimation_comparison.data_collection.preprocessor.linear_sample import LinearSampler
from estimation_comparison.data_collection.summary_stats import max_outside_middle_notch, autocorrelation_lag, \
    proportion_above_metric_cutoff
from estimation_comparison.database import BenchmarkDatabase
from estimation_comparison.model import Compressor, Estimator, Preprocessor, InputFile, IntermediateEstimationResult, \
    EstimationResult, LoadedData, BlockSummaryFunc, FileSummaryFunc, PreprocessedData


class Benchmark:
    def __init__(self, input_dir: List[str], output_dir: str, tags_csv: str):
        self._init_time = default_timer()
        self._tags_csv: Optional[pathlib.Path] = Path(tags_csv)
        self.data_locations = input_dir
        self.output_dir = output_dir
        self.database = BenchmarkDatabase(Path(self.output_dir) / "benchmark.sqlite")

        self._preprocessors: List[Preprocessor] = [
            Preprocessor(name="entire_file", instance=FlattenSampler()),
            Preprocessor(name="patch_random_25%", instance=PatchSampler(fraction=0.25)),
            Preprocessor(name="patch_random_50%", instance=PatchSampler(fraction=0.5)),
            Preprocessor(name="patch_random_75%", instance=PatchSampler(fraction=0.75)),
            Preprocessor(name="linear_random_25%", instance=LinearSampler(fraction=0.25)),
            Preprocessor(name="linear_random_50%", instance=LinearSampler(fraction=0.5)),
            Preprocessor(name="linear_random_75%", instance=LinearSampler(fraction=0.75)),
        ]

        self._block_summary_funcs: List[BlockSummaryFunc] = [
            BlockSummaryFunc(name="max_outside_middle_notch_64",
                             instance=max_outside_middle_notch,
                             parameters={"notch_width": 64}
                             ),
            BlockSummaryFunc(name="lag_1",
                             instance=autocorrelation_lag,
                             parameters={"lag": 1}
                             ),
            BlockSummaryFunc(name="lag_3",
                             instance=autocorrelation_lag,
                             parameters={"lag": 3}
                             ),
        ]

        self._block_summary_funcs += [BlockSummaryFunc(name=f"proportion_above_metric_cutoff_{x / 100}",
                                                       instance=proportion_above_metric_cutoff,
                                                       parameters={"cutoff": {x / 100}}
                                                       ) for x in range(5, 100, 5)]

        self._file_summary_funcs: List[FileSummaryFunc] = [
            FileSummaryFunc(name="mean",
                            instance=np.mean
                            ),
        ]

        self._estimators: List[Estimator] = [
            Estimator(
                name="autocorrelation_972",
                instance=Autocorrelation(
                    block_size=972
                ),
                summarize_block=True,
                summarize_file=True
            ),
            Estimator(name="bytecount_file", instance=ByteCount()),
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
        logging.info(f"Dask dashboard available: {self.client.dashboard_link}")

    def update_database(self):
        logging.info("Updating benchmark database compressor list")
        self.database.update_compressors(self._compressors)
        logging.info("Updating benchmark database estimator list")
        self.database.update_estimators(self._estimators)
        logging.info("Updating benchmark database preprocessor list")
        self.database.update_preprocessors(self._preprocessors)
        logging.info("Updating benchmark database block summary function lists")
        self.database.update_block_summary_funcs(self._block_summary_funcs)
        logging.info("Updating benchmark database file summary function lists")
        self.database.update_file_summary_funcs(self._file_summary_funcs)
        # logging.info("Updating benchmark database file hash list")
        # self.database.update_files(self.client, self.data_locations)
        # if self._tags_csv is not None:
        #     logging.info("Updating benchmark database file tags")
        #     self.database.update_tags(self._tags_csv)
        # logging.info("Updating benchmark database compression results")
        # self.database.update_compression_results(self.client, self._compressors)

    @staticmethod
    def _load_file(file: InputFile) -> LoadedData | None:
        try:
            with open(file.path, "rb") as f:
                data = f.read()
                if tiff_check(data):
                    data = tiff_decode(data)
                    return LoadedData(data=data, input_file=file)
                else:
                    return LoadedData(data=da.from_array(data, chunks=972), input_file=file)
        except OSError as e:
            logging.exception(f"Error reading {file}: {e}")

    @staticmethod
    def _preprocess_file(preprocessor: Preprocessor, data: LoadedData) -> PreprocessedData:
        return PreprocessedData(preprocessor.instance.run(data.data), input_file=data.input_file,
                                preprocessor=preprocessor)

    @staticmethod
    def _run_estimator(estimator: Estimator, bsf: BlockSummaryFunc, fsf: FileSummaryFunc,
                       ppd: PreprocessedData) -> IntermediateEstimationResult | None:
        try:
            return IntermediateEstimationResult.from_preprocessed_data(ppd, estimator.instance.run(ppd.data), estimator,
                                                                       bsf, fsf)
        except Exception as e:
            logging.exception(f"Error estimating {ppd.input_file}: {e}")

    @staticmethod
    def _run_block_summary(ier: IntermediateEstimationResult) -> IntermediateEstimationResult | None:
        try:
            if ier.block_summary_func is not None:
                memoized = functools.partial(ier.block_summary_func.instance, **(
                    ier.block_summary_func.parameters if ier.block_summary_func.parameters is not None else {}))
                print(f"ier {ier.result.shape}")
                ier.result = da.apply_along_axis(memoized, 0, ier.result)
            return ier
        except Exception as e:
            logging.exception(f"Error running {ier.block_summary_func} on {ier.input_file}: {e}")

    @staticmethod
    def _run_file_summary(ier: IntermediateEstimationResult) -> EstimationResult | None:
        try:
            if ier.file_summary_func is not None:
                return EstimationResult.from_intermediate_result(
                    ier, value=ier.file_summary_func.instance(ier.result, **(
                        ier.file_summary_func.parameters if ier.file_summary_func.parameters is not None else {})))
            return EstimationResult.from_intermediate_result(ier, ier.result)
        except Exception as e:
            logging.exception(f"Error running {ier.file_summary_func} on {ier.input_file}: {e}")

    def run(self):
        start_time = default_timer()
        completed_tasks = 0

        estimation_tasks = self.database.get_missing_estimation_results()

        for batch in itertools.batched(estimation_tasks[:10], 10):
            estimation_results = []
            for task in batch:
                loaded_file = self.client.submit(self._load_file, file=task.input_file)

                preprocessor = next(filter(lambda x: x.name == task.preprocessor_name, self._preprocessors))
                preprocessed = self.client.submit(self._preprocess_file, preprocessor=preprocessor,
                                                  data=loaded_file)

                estimator = next(filter(lambda x: x.name == task.estimator_name, self._estimators))
                try:
                    bsf = next(filter(lambda x: x.name == task.block_summary_func_name, self._block_summary_funcs))
                except StopIteration:
                    bsf = None
                try:
                    fsf = next(filter(lambda x: x.name == task.file_summary_func_name, self._file_summary_funcs))
                except StopIteration:
                    fsf = None
                estimated = self.client.submit(self._run_estimator, estimator=estimator, bsf=bsf,
                                               fsf=fsf, ppd=preprocessed)

                block_summarized = self.client.submit(self._run_block_summary, ier=estimated)

                estimation_results.append(self.client.submit(self._run_file_summary, ier=block_summarized))

            for future, result in as_completed(estimation_results, with_results=True):
                completed_tasks += 1
                logging.info(
                    f"{completed_tasks}/{len(estimation_tasks)} estimation tasks complete, {completed_tasks / len(estimation_tasks) * 100:.2f}%")
                try:
                    self.database.update_estimation_result(result.value.compute())
                except Exception as e:
                    logging.exception(f"Input file '{result.input_file.name}' raised exception\n\t{e}")

        logging.info(f"Estimation completed in {default_timer() - start_time:.3f} seconds")
        logging.info(f"Benchmark completed in {default_timer() - self._init_time:.3f} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", action="append")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    parser.add_argument("-l", "--limit-files", type=int, dest="file_limit", default=0)
    parser.add_argument("-o", "--output-dir", type=str, dest="output_dir", default="./benchmarks")
    parser.add_argument("-t", "--tags-csv", type=str, dest="tags_csv", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    benchmark = Benchmark(args.dir, args.output_dir, args.tags_csv)
    benchmark.update_database()

    benchmark.run()


if __name__ == "__main__":
    main()
