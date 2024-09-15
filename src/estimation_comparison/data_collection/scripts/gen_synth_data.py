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
import dataclasses
import logging
from pathlib import Path
from typing import Callable


@dataclasses.dataclass
class _SynthDataFileSpec:
    name: str
    func: Callable[[int], bytes]

    def gen(self, size: int) -> bytes:
        return self.func(size)


class SyntheticDataGenerator():
    def __init__(self, output_dir: Path):
        self.output_dir: Path = output_dir

        self.file_specs: [_SynthDataFileSpec] = [
            _SynthDataFileSpec("zeroes", lambda size: b"\x00" * size),
            _SynthDataFileSpec("count", lambda size: bytes([x % 256 for x in range(0, size)])),
            _SynthDataFileSpec("count_64", lambda size: bytes([x % 64 for x in range(0, size)])),
        ]

    def run(self):
        self._write_files()

    def _write_files(self):
        if not self.output_dir.exists():
            self.output_dir.mkdir()
        for file_spec in self.file_specs:
            with open(Path.cwd() / self.output_dir / file_spec.name, mode='wb') as f:
                f.write(file_spec.gen(1024 * 10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    parser.add_argument("-o", "--output_dir", type=Path, dest="output_dir", default="synthetic")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    sdg = SyntheticDataGenerator(args.output_dir)

    sdg.run()


if __name__ == "__main__":
    main()
