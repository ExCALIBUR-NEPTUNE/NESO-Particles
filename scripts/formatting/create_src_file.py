"""
This is a helper script to create source files for an input header file.
"""

import sys
import os

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Expected a path to a header file.")
        quit()

    input_file = os.path.abspath(sys.argv[1])
    root = os.path.abspath(os.path.join(__file__, "../../.."))
    assert (
        os.path.commonpath((root, input_file)) == root
    ), "File is not in NESO-Particles?"

    input_file = os.path.relpath(input_file, start=root)
    header_to_include = os.path.relpath(input_file, start="include")
    output_file = os.path.relpath(
        input_file, start=os.path.join("include", "neso_particles")
    )
    output_file = os.path.join(
        root, "src", "".join(output_file.split(".")[:-1] + [".cpp"])
    )

    if os.path.exists(output_file):
        print("Output file already exists:")
        print(output_file)
        quit()

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    print(header_to_include)
    print(input_file)
    print(output_file)
    print(output_dir)

    file_contents = f"""#include <{header_to_include}>

namespace NESO::Particles {{

}}
"""

    with open(output_file, "w+") as fh:
        fh.write(file_contents)
