# This script formats the source files in NESO. This script should be ran from
# the root of the git repository. This script should be run as:
#
#   bash format_all.sh
#
# to format all source files.

if [[ -f .clang-format && -f .cmake-format ]]; then
    # clang-format
    find ./src ./include ./test ./solvers -iname \*.hpp -o -iname \*.cpp | xargs clang-format-18 -i
    # cmake-format
    cmake-format -c .cmake-format -i CMakeLists.txt # Do this one on it's own so we don't go into build etc
    find ./src ./include ./test ./solvers -iname CMakeLists.txt | xargs cmake-format  -c .cmake-format -i
    # black
    find ./python -iname \*.py | xargs black
    exit 0;
else
    echo "ERROR: The files .clang-format and .cmake-format do not exist. Please
       check this script is executed from the root directory of the git
       repository."
    exit 1;
fi
