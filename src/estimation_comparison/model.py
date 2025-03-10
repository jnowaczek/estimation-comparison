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
from dataclasses import dataclass
from typing import Self, Callable, Optional

import numpy as np

from estimation_comparison.data_collection.compressor.general import GeneralCompressorBase
from estimation_comparison.data_collection.compressor.image import ImageCompressorBase
from estimation_comparison.data_collection.estimator import EstimatorBase
from estimation_comparison.data_collection.preprocessor import BaseSampler


# Benchmark config classes
# Also used in database model

@dataclass
class Estimator:
    name: str
    instance: EstimatorBase
    summarize_block: bool = False
    summarize_file: bool = False


@dataclass
class Compressor:
    name: str
    instance: GeneralCompressorBase | ImageCompressorBase


@dataclass
class Preprocessor:
    name: str
    instance: BaseSampler[np.ndarray]


@dataclass
class BlockSummaryFunc:
    name: str
    instance: Callable
    parameters: Optional[dict] = None


@dataclass
class FileSummaryFunc:
    name: str
    instance: Callable
    parameters: Optional[dict] = None


# Database model classes

@dataclass
class InputFile:
    hash: str
    path: str
    name: str
    size_bytes: int


@dataclass
class CompressionTask:
    input_file: InputFile
    compressor_name: str


@dataclass
class EstimationTask:
    input_file: InputFile
    preprocessor_name: str
    estimator_name: str
    block_summary_func_name: Optional[str]
    file_summary_func_name: Optional[str]


# Result model classes

@dataclass
class LoadedData:
    data: any
    input_file: InputFile


@dataclass
class CompressionResult:
    input_file: InputFile
    compressor: Compressor
    compressed_size_bytes: int


@dataclass
class PreprocessedData:
    data: np.ndarray
    input_file: InputFile
    preprocessor: Preprocessor

    @classmethod
    def from_loaded_data(cls, ld: LoadedData, preprocessor: Preprocessor) -> Self:
        return PreprocessedData(data=ld.data, input_file=ld.input_file, preprocessor=preprocessor)


@dataclass
class IntermediateEstimationResult:
    result: np.ndarray | int | float
    input_file: InputFile
    preprocessor: Preprocessor
    estimator: Estimator
    block_summary_func: Optional[BlockSummaryFunc]
    file_summary_func: Optional[FileSummaryFunc]

    @classmethod
    def from_preprocessed_data(cls, pd: PreprocessedData, result: np.ndarray, estimator: Estimator,
                               block_summary_func: BlockSummaryFunc,
                               file_summary_func: FileSummaryFunc) -> Self:
        return IntermediateEstimationResult(result=result, input_file=pd.input_file, preprocessor=pd.preprocessor,
                                            estimator=estimator, block_summary_func=block_summary_func,
                                            file_summary_func=file_summary_func)


@dataclass
class EstimationResult:
    value: int | float
    input_file: InputFile
    preprocessor: Preprocessor
    estimator: Estimator
    block_summary_func: Optional[BlockSummaryFunc]
    file_summary_func: Optional[FileSummaryFunc]

    @classmethod
    def from_intermediate_result(cls, ir: IntermediateEstimationResult, value: int | float) -> Self:
        return EstimationResult(value=value,
                                input_file=ir.input_file,
                                preprocessor=ir.preprocessor, estimator=ir.estimator,
                                block_summary_func=ir.block_summary_func,
                                file_summary_func=ir.file_summary_func)
