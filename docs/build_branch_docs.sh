#!/bin/bash

# create the output dir where we collect doc versions
OUTPUT_DIR=$(pwd)/build
mkdir -p ${OUTPUT_DIR}

# clone the repo into a temporary place
REPO=https://github.com/ExCALIBUR-NEPTUNE/NESO-Particles.git
mkdir /tmp/repo-checkout
cd /tmp/repo-checkout
git clone $REPO
cd NESO-Particles/docs

# checkout each tag and the main branch, build the docs for that version
for BX in `git tag -l --sort=refname v*` main
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



