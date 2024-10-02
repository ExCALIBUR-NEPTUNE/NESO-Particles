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
$ python profile_region_plotting.py -s <start_time> -e <end_time> *.json

# Run with -h for a complete set of options.
```

