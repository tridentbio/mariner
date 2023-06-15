#!/bin/bash

mkdir -p pydeps_data

for module_name in "$@"
do
    pydeps ./${module_name}/ --noshow -o "pydeps_data/${module_name}.svg"
done