"""
Call this tool like

python3 include_checker.py <include_root_directory> <path to compile_commands.json>

"""

import sys
import os
import glob
import tempfile
import subprocess
import json
import shlex

if __name__ == "__main__":

    include_root_directory = os.path.abspath(sys.argv[1])

    def remove_arg(command, arg):
        command = command.split(" " + arg + " ")
        command = command[0].split() + command[1].split()[1:]
        command = " ".join(command)
        return command

    compile_command = None
    with open(sys.argv[2]) as fh:
        all_compile_commands = fh.read()
        json_compile_commands = json.loads(all_compile_commands)
        compile_command = json_compile_commands[0]["command"]
        compile_command = remove_arg(compile_command, "-c")
        compile_command = remove_arg(compile_command, "-o")

    compile_command = shlex.split(compile_command)
    compile_command += ["-o", "foo.o"]

    files = glob.glob(
        os.path.join(include_root_directory, "**", "*.hpp"), recursive=True
    )

    print(" ".join(compile_command))

    for fx in files:

        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = os.path.join(tmpdirname, "main.cpp")
            with open(filename, "w") as fh:
                fh.write(
                    """
#include "{HEADER}"
int main(int argc, char ** argv) {{
    return 0;
}}
""".format(
                        HEADER=fx
                    )
                )
            compile_command_full = compile_command + ["-c", filename]
            process = subprocess.Popen(
                compile_command_full,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdirname,
            )
            stdout, stderr = process.communicate()

            print("=" * 80)
            print(fx)
            print("." * 80)
            print(stdout.decode(sys.getdefaultencoding()))
            print(stderr.decode(sys.getdefaultencoding()))
