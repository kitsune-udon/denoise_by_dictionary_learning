#!/bin/env sh

function main () {
  python make_dictionary.py $1 && \
  python visualize_dictionary.py $1 && \
  python denoise.py $1
}

main $1
