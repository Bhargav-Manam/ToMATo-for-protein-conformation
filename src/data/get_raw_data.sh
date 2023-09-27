#!/usr/bin/env bash

# Get data
cd ../../data/raw/ && \
wget https://www-sop.inria.fr/abs/teaching/centrale-FGMDA/exam_mathieu/aladip.zip && \
unzip aladip.zip && \
mv aladip/* . && \
rm -r aladip aladip.zip

# Pre-process

dos2unix dihedral.xyz
sed -i 's/^[[:space:]]*//' dihedral.xyz
sed -i 's/ \+/ /g' dihedral.xyz