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
    }

    colour_mapper = ColourMapper()

    print(f"Found {len(files)} source files.")

    for fx in files:
        data = json.loads(open(fx).read())
        rank = data["rank"]
        for rx in data["regions"]:
            time_start = rx[2]
            time_end = rx[3]
            if (time_start >= cutoff_start) and (time_end <= cutoff_end):
                dd["rank"].append(rank)
                name = rx[0] + ":" + rx[1]
                dd["name"].append(name)
                dd["time_start"].append(time_start)
                dd["time_end"].append(time_end)
                dd["time_elapsed_plot"].append(time_end - time_start)
                dd["time_elapsed"].append(time_end - time_start)
                dd["colour"].append(
                    px.colors.qualitative.Dark24[colour_mapper.get(name)]
                )

    df = pd.DataFrame.from_dict(dd)
    print(df)

    labels = {
        "time_elapsed_plot": "Time",
        "time_start": "Start Time",
        "time_end": "End Time",
        "time_elapsed": "Time Elapsed",
    }

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
            "colour": False,
        },
        color="name",
        barmode="overlay",
        # barmode="group",
        # barmode="relative",
        color_discrete_sequence=px.colors.qualitative.Dark24,
        hover_name="name",
        labels=labels,
    )

    fig.show()
