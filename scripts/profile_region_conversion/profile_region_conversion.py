"""
This is a Python3 program to convert event data output by the NESO-Particles
ProfileMap into different file formats.
"""

import json
import argparse
import sys
import os


def print_file_info(filename):
    """
    Print information about a file, e.g. start and end times.

    :param filename: File name to collect and print information for.
    """
    json_contents = json.loads(open(filename).read())
    rank = int(json_contents["rank"])
    regions = json_contents["regions"]
    events = []

    max_time = 0.0
    min_time = sys.float_info.max
    num_events = 0

    for ix, region in enumerate(regions):
        time_start = region[2]
        time_end = region[3]
        max_time = max(time_end, max_time)
        min_time = min(time_start, min_time)
        num_events += 1

    print(
        os.path.basename(filename),
        "First event time:",
        min_time,
        "Last event time:",
        max_time,
    )


def convert_file_trace_event_format(filename, cutoff_start, cutoff_end):
    """
    Convert the events in a file to the Trace Event Format.

    :param filename: File to load source events from.
    :param cutoff_start: Time in seconds to use as the start of the window of interest.
    :param cutoff_end: Time in seconds to use as the end of the window of interest.
    :returns: List of dictonary events in the Trace Event Format.
    """
    json_contents = json.loads(open(filename).read())
    rank = int(json_contents["rank"])
    regions = json_contents["regions"]
    events = []

    for ix, region in enumerate(regions):
        name = region[0] + ":" + region[1]

        time_start = region[2]
        time_end = region[3]

        if (time_start >= cutoff_start) and (time_end <= cutoff_end):

            ts_start = region[2] * 1e6
            ts_end = region[3] * 1e6
            ts_duration = max(1, ts_end - ts_start)

            events.append(
                {
                    "name": name,
                    "cat": "PERF",
                    "ph": "X",
                    "ts": ts_start,
                    "dur": ts_duration,
                    "pid": rank,
                    "tid": rank,
                    "args": {},
                }
            )

    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="profile_region_conversion",
        description="Converts NESO-Particles ProfileMap JSON output to other formats",
    )
    parser.add_argument(
        "-f",
        nargs="?",
        default="trace_event_format",
        help="Output file format. Default Trace Event Format.",
    )
    parser.add_argument("-o", nargs="?", help="Output file name.")
    parser.add_argument(
        "-s", nargs="?", help="Optional start time.", type=float, default=-1.0
    )
    parser.add_argument(
        "-e",
        nargs="?",
        help="Optional end time.",
        type=float,
        default=sys.float_info.max,
    )
    parser.add_argument(
        "-i", action="store_true", help="Print information about files."
    )

    if len(sys.argv) == 1:
        parser.print_help()
    args, json_files = parser.parse_known_args(sys.argv[1:])

    sys.tracebacklimit = 0
    assert (
        args.f == "trace_event_format"
    ), "Trace Event Format is currently the only accepted format."
    assert len(json_files) > 0, "No input files passed."
    sys.tracebacklimit = 1000

    if args.i:
        for filex in json_files:
            print_file_info(filex)

        quit()

    output_json = {
        "traceEvents": [],
        "displayTimeUnit": "ms",
    }

    for filex in json_files:
        output_json["traceEvents"] += convert_file_trace_event_format(
            filex, args.s, args.e
        )

    with open(args.o, "w") as fh:
        fh.write(json.dumps(output_json))
