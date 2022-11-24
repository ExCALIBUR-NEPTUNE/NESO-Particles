#!/bin/bash

# create the output dir where we collect doc versions
OUTPUT_DIR=$(pwd)/build
mkdir -p ${OUTPUT_DIR}

# determine the branches from the switcher json (could also list tags instead)
DOCSVERSIONS=$(python3 -c "import json; print(' '.join([fx['version'] for fx in json.loads(open('./switcher.json').read())]))")
echo $DOCSVERSIONS

# clone the repo into a temporary place
REPO=https://github.com/ExCALIBUR-NEPTUNE/NESO-Particles.git
mkdir /tmp/repo-checkout
cd /tmp/repo-checkout
git clone $REPO
cd NESO-Particles/docs

# checkout each version to build and build the docs for that version in tmp
for BX in $DOCSVERSIONS
do
    echo $BX
    echo $(pwd)

    # checkout a version and build the docs for it
    git checkout $BX
    echo "$BX" > ./sphinx/source/docs_version
    cat ./sphinx/docs_version
    make

    # create a directory for this version in the global output directory
    BRANCH_OUTPUT=${OUTPUT_DIR}/$BX
    mkdir -p ${BRANCH_OUTPUT}
    # copy the docs for this version to the global output directory
    mv build/* ${BRANCH_OUTPUT}
done



