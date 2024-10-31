"""
Helper script for plotting ProfileRegions. Usage like:

python profile_region_plotting.py -s <start_time> -e <end_time> *.json
"""

import json
import os
import sys
import pandas as pd
import plotly.express as px
import argparse


class ColourMapper:
    def __init__(self):
        self.num_colours = 0
        self.names = {}

    def get(self, name):
        if name in self.names:
            return self.names[name]
        else:
            n = self.num_colours
            self.names[name] = n
            self.num_colours += 1
            return n


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=f"""Plot regions from NESO-Particles ProfileRegion. For example:
    python {os.path.basename(sys.argv[0])} -s <start_time> -e <end_time> *.json"""
    )

    parser.add_argument(
        "-s",
        type=float,
        default=-sys.float_info.max,
        help="Specify the start time for region reading and plotting. By Default the start of all regions will be used.",
    )
    parser.add_argument(
        "-e",
        type=float,
        default=sys.float_info.max,
        help="Specify the end time for region reading and plotting. By default the end of all regions will be used.",
    )
    parser.add_argument(
        "-g",
        "--group",
        action="store_true",
        help="Group regions by name in the y-direction. Only recomended with small numbers of input files.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only compute metrics, do not try and plot.",
    )
    parser.add_argument(
        "json_files",
        nargs=argparse.REMAINDER,
        help="JSON files to parse and plot.",
    )

    args_all = parser.parse_known_args()
    args = args_all[0]
    files = args_all[0].json_files
    if len(files) < 1:
        print("Did not find any JSON files to process.")
        exit(-1)

    cutoff_start = args.s
    cutoff_end = args.e

    dd = {
        "rank": [],
        "name": [],
        "time_start": [],
        "time_end": [],
        "time_elapsed_plot": [],
        "time_elapsed": [],
        "colour": [],
        "bandwidth": [],
        "flops": [],
        "num_bytes": [],
        "num_flops": [],
    }

    colour_mapper = ColourMapper()

    print(f"Found {len(files)} source files.")

    max_name_length = 0

    for fx in files:
        data = json.loads(open(fx).read())
        rank = data["rank"]
        for rx in data["regions"]:
            time_start = rx[2]
            time_end = rx[3]
            level = 0
            if len(rx) > 4:
                level = rx[4]
            num_bytes = 0
            num_flops = 0
            if len(rx) > 6:
                num_bytes = rx[5]
                num_flops = rx[6]
            if (time_start >= cutoff_start) and (time_end <= cutoff_end):
                dd["rank"].append(rank)
                name = rx[0] + ":" + rx[1]
                dd["name"].append(name)
                max_name_length = max(max_name_length, len(name))
                dd["time_start"].append(time_start)
                dd["time_end"].append(time_end)
                dd["time_elapsed_plot"].append(time_end - time_start)
                time_elapsed = time_end - time_start
                dd["time_elapsed"].append(time_elapsed)
                dd["colour"].append(
                    px.colors.qualitative.Dark24[colour_mapper.get(name) % 24]
                )
                bandwidth = 0.0
                flops = 0.0
                if time_elapsed > 0.0:
                    bandwidth = num_bytes / (time_elapsed * 1.0e9)
                    flops = num_flops / (time_elapsed * 1.0e9)
                dd["bandwidth"].append(bandwidth)
                dd["flops"].append(flops)
                dd["num_bytes"].append(num_bytes)
                dd["num_flops"].append(num_flops)

    ranks = sorted(set(dd["rank"]))
    df = pd.DataFrame.from_dict(dd)
    dd = None

    dd_metrics = {"name": [], "flops": [], "bandwidth": []}

    for rank in ranks:
        print(40 * "-", "Rank:", rank, 40 * "-")
        df_rank = df[df["rank"] == rank]
        keys = set(df_rank["name"])
        for keyx in keys:
            df_key = df_rank[df_rank["name"] == keyx].sum()
            num_bytes = float(df_key["num_bytes"])
            num_flops = float(df_key["num_flops"])
            time_elapsed = float(df_key["time_elapsed"])
            if ((num_bytes > 0.0) or (num_flops > 0.0)) and (
                time_elapsed > 0.0
            ):
                bandwidth = num_bytes / (time_elapsed * 1.0e9)
                flops = num_flops / (time_elapsed * 1.0e9)
                name = keyx.ljust(max_name_length)
                print(
                    "{} {:12.2e} GFLOP/s {:12.2e} GB/s".format(
                        name, flops, bandwidth
                    )
                )
                dd_metrics["name"].append(name)
                dd_metrics["flops"].append(flops)
                dd_metrics["bandwidth"].append(bandwidth)

    names = set(dd_metrics["name"])
    df_metrics = pd.DataFrame.from_dict(dd_metrics)
    dd_metrics = None

    print(40 * "=", "Totals", 41 * "=")
    for name in names:
        df_name = df_metrics[df_metrics["name"] == name].sum()
        print(
            "{} {:12.2e} GFLOP/s {:12.2e} GB/s".format(
                name, df_name["flops"], df_name["bandwidth"]
            )
        )

    if not args.metrics_only:
        labels = {
            "time_elapsed_plot": "Time",
            "time_start": "Start Time",
            "time_end": "End Time",
            "time_elapsed": "Time Elapsed",
            "bandwidth": "GB/s",
            "flops": "GFLOP/s",
        }

        barmode = "group" if args.group else "overlay"

        fig = px.bar(
            df,
            x="time_elapsed_plot",
            base="time_start",
            y="rank",
            orientation="h",
            hover_data={
                "name": True,
                "time_start": True,
                "time_end": True,
                "time_elapsed_plot": False,
                "time_elapsed": True,
                "bandwidth": True,
                "flops": True,
                "colour": False,
            },
            color="name",
            barmode=barmode,
            color_discrete_sequence=px.colors.qualitative.Dark24,
            hover_name="name",
            labels=labels,
        )

        fig.show()
