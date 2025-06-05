import json
import sys
import subprocess

if __name__ == "__main__":

    files = sys.argv[1:]

    counts = {}

    for fx in files:
        data = json.loads(open(fx).read())
        names_map_mangled = data["index_to_mangled_names"]
        names_map = {}
        for index, mangled_name in names_map_mangled.items():
            index = int(index)
            filt = (
                subprocess.check_output(["c++filt", "-t", mangled_name])
                .decode(sys.stdout.encoding)
                .strip()
            )
            filt = "".join(filt.split())
            filt = "".join(filt.split("NESO::Particles::"))
            filt = filt.replace("double", "REAL")
            filt = filt.replace("long", "INT")
            names_map[index] = filt

        calls = data["loop_count_to_args_count"]

        for callx in calls:
            args = callx[:-1]
            count = callx[-1]

            key = tuple([names_map[ax] for ax in args])

            if not key in counts:
                counts[key] = 0
            counts[key] += count

    keys = [kx for kx in counts.keys()]
    keys.sort(key=lambda x: counts[x])

    for key in keys:

        statement = (
            "extern template class ParticleLoopArgs<" + ", ".join(key) + ">;"
        )

        print("{:<8d}".format(counts[key]), statement)
