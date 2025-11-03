# Introduction

This is a utility to convert the event format output by `ProfileMap` into different formats, e.g. Trace Event Format.
Currently this tool only has dependencies in the Python standard library.

```
# Run the script, the start and end times are optional.
$ python profile_region_conversion.py -s <start_time> -e <end_time> -o <output_file_name> *.json

# Run with -h for a complete set of options.
```

