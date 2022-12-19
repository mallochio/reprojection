#! /bin/sh
#
# batch_flip.sh
# Copyright (C) 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.
#

for f in "$1"/*.{jpg,png}; do
  [ -f "$f" ] || continue
  base=$(basename "$f")
  convert -flop "$f" "$1/${base%.*}.${base##*.}"
done

