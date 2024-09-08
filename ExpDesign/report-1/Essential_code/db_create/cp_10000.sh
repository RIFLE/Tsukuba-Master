#!/bin/bash

# define source and target directories
src="./Normal_512_10000"
target="./2000_Normal"

# iterate over first 2000 .pgm files
for i in {1..10000}
do
   cp "${src}/${i}.pgm" "${target}/"
done

echo "Done";
