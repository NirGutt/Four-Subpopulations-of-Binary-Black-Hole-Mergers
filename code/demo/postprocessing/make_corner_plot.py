#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import bilby


parser = argparse.ArgumentParser(description="Create a corner plot from a Bilby result.")
parser.add_argument("--result_file",default='demo/outdir/result/tiny_demo_result.hdf5', help="Bilby .hdf5 or .json result file")
args = parser.parse_args()

result_file = Path(args.result_file)
output_file = result_file.with_name(f"{result_file.stem}_corner.png")

result = bilby.result.read_in_result(filename=str(result_file))

plot_options = {"filename": str(output_file),
                 "parameters": ['mpp_1','mpp_2']
               }
result.plot_corner(**plot_options)

print(f"Corner plot saved to: {output_file}")
