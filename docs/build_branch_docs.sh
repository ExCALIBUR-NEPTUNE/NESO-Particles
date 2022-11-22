#!/bin/bash

OUTPUT_DIR=$(pwd)/build
mkdir -p ${OUTPUT_DIR}
BRANCHES=$(python3 -c "import json; print(' '.join([fx['version'] for fx in json.loads(open('./switcher.json').read())]))")
echo $BRANCHES
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
    echo "$BX" > ./sphinx/source/docs_version
    cat ./sphinx/docs_version
    make
    BRANCH_OUTPUT=${OUTPUT_DIR}/$BX
    mkdir -p ${BRANCH_OUTPUT}
    mv build/* ${BRANCH_OUTPUT}
done



