# Dependency Installation

The requirements for this script are contained in the `requirement.txt` file and can be installed into a Python virtual environment as follows:

```
# Create and activate a virtual environment
$ python3 -m venv profile_region_plotting_env
$ source profile_region_plotting_env/bin/activate

# Install the dependencies into the virtual environment
(profile_region_plotting_env)$ pip install -r requirements.txt
# pip output omitted

# Run the script
$ python profile_region_plotting.py -h
usage: profile_region_plotting.py [-h] [-s S] [-e E] ...

Plot regions from NESO-Particles ProfileRegion. For example: python profile_region_plotting.py -s <start_time>
-e <end_time> *.json

positional arguments:
  json_files  JSON files to parse and plot.

options:
  -h, --help  show this help message and exit
  -s S        Specify the start time for region reading and plotting. By Default the start of all regions will
              be used.
  -e E        Specify the end time for region reading and plotting. By default the end of all regions will be
              used.
```

