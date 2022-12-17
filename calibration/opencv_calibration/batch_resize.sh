#! /bin/sh
#
# batch_resize.sh
# Copyright (C) 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.
#

for f in "$1"/*.{jpg,png}; do
  [ -f "$f" ] || continue
  base=$(basename "$f")
  convert -resize 1280x720 "$f" "$1/${base%.*}.${base##*.}"
done

