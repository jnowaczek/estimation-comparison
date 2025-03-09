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
from typing import NamedTuple, List, Self, Callable, Optional

import numpy as np

from estimation_comparison.data_collection.compressor.general import GeneralCompressorBase
from estimation_comparison.data_collection.compressor.image import ImageCompressorBase
from estimation_comparison.data_collection.estimator import EstimatorBase
from estimation_comparison.data_collection.preprocessor import BaseSampler

InputFile = NamedTuple("InputFile", [("hash", str), ("path", str), ("name", str), ("size_bytes", int)])

Ratio = NamedTuple("Ratio", [("hash", str), ("algorithm", str), ("ratio", float)])
CompressionResult = NamedTuple("CompressionResult", [("hash", str), ("algorithm", str), ("size_bytes", int)])
Metric = NamedTuple("Metric", [("hash", str), ("estimator", str), ("metric", bytes)])

FriendlyRatio = NamedTuple("Ratio", [("file_name", str), ("algorithm", str), ("ratio", float)])
FriendlyMetric = NamedTuple("Metric", [("file_name", str), ("preprocessor", str), ("estimator", str), ("metric", any)])


@dataclass
class EstimationTask:
    hash: str
    path: str
    name: str
    size_bytes: int
    preprocessor_name: str
    estimator_name: str
    block_summary_func_name: Optional[str]
    file_summary_func_name: Optional[str]


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


@dataclass
class LoadedData:
    data: np.ndarray
    completed_stages: List[str]
    input_file: InputFile


@dataclass
class IntermediateEstimationResult:
    data: np.ndarray
    completed_stages: List[str]
    input_file: InputFile
    preprocessor: Preprocessor
    block_summary: Optional[BlockSummaryFunc]
    file_summary: Optional[FileSummaryFunc]


@dataclass
class EstimationResult:
    value: int | float
    completed_stages: List[str]
    input_file: InputFile
    preprocessor: Preprocessor
    estimator: Estimator
    block_summary: Optional[BlockSummaryFunc]
    file_summary: Optional[FileSummaryFunc]

    @classmethod
    def from_intermediate_result(cls, ir: IntermediateEstimationResult, value: int | float,
                                 estimator: Estimator) -> Self:
        return EstimationResult(value=value, completed_stages=ir.completed_stages + [estimator.name],
                                input_file=ir.input_file,
                                preprocessor=ir.preprocessor, estimator=estimator, block_summary=ir.block_summary,
                                file_summary=ir.file_summary)
