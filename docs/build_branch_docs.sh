#!/bin/bash

OUTPUT_DIR=$(pwd)/build
mkdir -p ${OUTPUT_DIR}
BRANCHES="dev-docs"
REPO=https://github.com/ExCALIBUR-NEPTUNE/NESO-Particles.git

mkdir /tmp/repo-checkout
cd /tmp/repo-checkout
git clone $REPO
cd NESO-Particles/docs

for BX in $BRANCHES
do
    echo $BX
    echo $(pwd)
    git checkout $BX
    make
    BRANCH_OUTPUT=${OUTPUT_DIR}/$BX
    mkdir -p ${BRANCH_OUTPUT}
    mv build/* ${BRANCH_OUTPUT}
done



