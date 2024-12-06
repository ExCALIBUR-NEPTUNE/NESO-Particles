*********
Profiling
*********

Built-in Profiling
==================

Internally NP can record the time taken for expensive operations and internal ParticleLoops.
The time taken for user written loops can also be recorded.
For these recorded times to make much sense it is beneficial to given ParticleLoops a meaningful name.
Furthermore code regions, which do not have to be ParticleLoops, can also be added to the set of recorded regions by using the ProfileRegion class.

.. warning::
    Profiling of regions is disabled unless `NESO_PARTICLES_PROFILING_REGION` is defined.


To enable built in profiling of regions the preprocessor variable `NESO_PARTICLES_PROFILING_REGION` must be defined. 
If this variable is not defined then the profiling functions and methods become no-ops.
This variable can be defined by compiler flags or defined before including NP as follows.

.. code-block:: cpp

    #define NESO_PARTICLES_PROFILING_REGION
    #include <neso_particles.hpp>


The `ProfileRegion` class records the time taken for a specific piece of code. 
These regions are recorded within a `ProfileMap` instance. 
There is a `ProfileMap` member within the `SYCLTarget` type which records the regions for ParticleLoops which execute on that compute device.

In the following code snippet we demonstrate how to profile ParticleLoops and user defined regions.
Finally this profiling data is written to a JSON file by each rank. 
The write operation is not collective and users may choose to only write the data from specific ranks.

.. literalinclude:: ../example_sources/example_profile_regions.hpp
   :language: cpp
   :caption: Example of using ProfileRegion and writing events to disk.


Plotting ProfileRegions
=======================

We provide a helper script `scripts/profile_region_plotting/profile_region_plotting.py` to aid plotting the regions written to the JSON file.
The requirements for this script are contained in the `requirement.txt` file and can be installed into a Python virtual environment as follows:
::

    # Create and activate a virtual environment
    $ python3 -m venv profile_region_plotting_env
    $ source profile_region_plotting_env/bin/activate

    # Install the dependencies into the virtual environment
    (profile_region_plotting_env)$ pip install -r requirements.txt
    # pip output omitted

    # Run the script
    $ python profile_region_plotting.py -s <start_time> -e <end_time> *.json

    # Run with -h for a complete set of options.


The script plots time on the x-axis and MPI rank on the y-axis.
On launch all recorded events, within the specified time window, are plotted.
To simplify the view double click on an item in the legend on the right hand side to focus only on regions with that name.
Other regions can then be added to the view one-by-one by clicking on them in the legend.




