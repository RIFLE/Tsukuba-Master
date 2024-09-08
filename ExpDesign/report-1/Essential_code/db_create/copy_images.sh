#!/bin/bash

source_dir="./10000_NORMAL" # Directory that contains the images
destination_base_dir="." # Base directory where the images will be stored

for i in {1..10}; do
  start_image=$(((i-1)*1000+1))
  end_image=$((i*1000))
  
  destination_dir="${destination_base_dir}/${i}"
  mkdir -p "$destination_dir"

  for image in $(seq $start_image $end_image); do
    cp "${source_dir}/${image}.pgm" "$destination_dir"
  done
done

