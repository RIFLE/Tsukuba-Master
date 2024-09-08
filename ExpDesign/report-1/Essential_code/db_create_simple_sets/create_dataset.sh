#!/bin/bash

# Assign directory names at the top of the script

DATA_QUANTITY=10000 # Make it {5,2}|DATA_QUANTITY 

ALGO_XX="./10000/HUGO-25/conforming_10000/10000_HUGO-25"
MASKS_ALGO_XX="./10000/HUGO-25/conforming_10000/10000_MASKS-HUGO-25"

NORMAL="./10000/10000_NORMAL_conforming"
MASKS_EMPTY="./10000/10000_EMPTY_conforming"

TRAINING_IMAGES="./HUGO_25_10000/training/images"
TRAINING_MASKS="./HUGO_25_10000/training/masks"

TEST_IMAGES="./HUGO_25_10000/testing/images"
TEST_MASKS="./HUGO_25_10000/testing/masks"


##########2000_HUGO_25#######
#ALGO_XX="./conforming_2000/2000_HUGO-25"
#MASKS_ALGO_XX="./conforming_2000/2000_MASKS-HUGO-25"

#NORMAL="./conforming_2000/2000_NORMAL"
#MASKS_EMPTY="./conforming_2000/empty_mask_2000"

#TRAINING_IMAGES="./2000_HUGO_25/training/images"
#TRAINING_MASKS="./2000_HUGO_25/training/masks"

#TEST_IMAGES="./2000_HUGO_25/testing/images"
#TEST_MASKS="./2000_HUGO_25/testing/masks"


#########WOW_10_ALL#########
#ALGO_XX="./conforming/WOW_10_ALL"
#MASKS_ALGO_XX="./conforming/MASK-WOW_10_ALL"

#NORMAL="./conforming/NORMAL_ALL"
#MASKS_EMPTY="./conforming/empty_mask"

#TRAINING_IMAGES="./WOW_10_exp/training/images"
#TRAINING_MASKS="./WOW_10_exp/training/masks"

#TEST_IMAGES="./WOW_10_exp/testing/images"
#TEST_MASKS="./WOW_10_exp/testing/masks"

#########WOW_25_ALL#########
#ALGO_XX="./conforming/WOW_25_ALL"
#MASKS_ALGO_XX="./conforming/MASK-WOW_25_ALL"

#NORMAL="./conforming/NORMAL_ALL"
#MASKS_EMPTY="./conforming/empty_mask"

#TRAINING_IMAGES="./WOW_25_exp/training/images"
#TRAINING_MASKS="./WOW_25_exp/training/masks"

#TEST_IMAGES="./WOW_25_exp/testing/images"
#TEST_MASKS="./WOW_25_exp/testing/masks"


#########HUGO_25_ALL#########
#ALGO_XX="./conforming/HUGO_25_ALL"
#MASKS_ALGO_XX="./conforming/MASK-HUGO_25_ALL"

#NORMAL="./conforming/NORMAL_ALL"
#MASKS_EMPTY="./conforming/empty_mask"

#TRAINING_IMAGES="./HUGO_25_exp/training/images"
#TRAINING_MASKS="./HUGO_25_exp/training/masks"

#TEST_IMAGES="./HUGO_25_exp/testing/images"
#TEST_MASKS="./HUGO_25_exp/testing/masks"


# Confirm with the user
echo "The script will run with the following directories:"
echo "ALGO_XX = $ALGO_XX"
echo "MASKS_ALGO_XX = $MASKS_ALGO_XX"
echo "NORMAL = $NORMAL"
echo "MASKS_EMPTY = $MASKS_EMPTY"
echo "TRAINING_IMAGES = $TRAINING_IMAGES"
echo "TRAINING_MASKS = $TRAINING_MASKS"
echo "TEST_IMAGES = $TEST_IMAGES"
echo "TEST_MASKS = $TEST_MASKS"
echo "Do you want to proceed? (yes/no)"

read answer

if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Proceeding..."
else
    echo "Aborting..."
    exit 1
fi

# Checking directories
for dir in $ALGO_XX $MASKS_ALGO_XX $NORMAL $MASKS_EMPTY
do
    if [ ! -d "$dir" ]; then
        echo "Directory $dir does not exist. Aborting..."
        exit 1
    fi

    npz_files=$(find $dir -type f -name "*.npz" | wc -l)
    if [ "$npz_files" -ne $DATA_QUANTITY ]; then
        echo "Directory $dir does not contain $DATA_QUANTITY .npz files. Aborting..."
        exit 1
    fi
done

# EASY EXAMPLE

# Training set
##for i in {1..225}
#for i in {1..950}
#do
#   cp "$ALGO_XX/$i.npz" "$TRAINING_IMAGES/$i.npz"
#   cp "$MASKS_ALGO_XX/$i.npz" "$TRAINING_MASKS/$i.npz"
#done
#
##for i in {1..225}
#for i in {1..950}
#do
#   j=$((i + 950))
#   cp "$NORMAL/$i.npz" "$TRAINING_IMAGES/$j.npz"
#   cp "$MASKS_EMPTY/$i.npz" "$TRAINING_MASKS/$j.npz"
#done
#
## Test set
##for i in {1..25}
#for i in {1..50}
#do
##  j=$((i + 450))
#   j=$((i + 1900))
#   cp "$ALGO_XX/$i.npz" "$TEST_IMAGES/$j.npz"
#   cp "$MASKS_ALGO_XX/$i.npz" "$TEST_MASKS/$j.npz"
#done
#
##for i in {1..25}
#for i in {1..50}
#do
#   j=$((i + 1950))
#   cp "$NORMAL/$i.npz" "$TEST_IMAGES/$j.npz"
#   cp "$MASKS_EMPTY/$i.npz" "$TEST_MASKS/$j.npz"
#done
#
#echo "DONE."; exit 0;


# SUPER-AVTOMATIZACIJA | NE TROGAT` ESLI NE PONIMAESH SHELL-SKRIPT I TEORIJU ARIFMETIKI ILI PC SGORIT

# i.e. Image quantity = 10000

TRAIN_PERCENT=95  # percentage for training data
TRAIN_QUANTITY=$(echo "scale=0; $DATA_QUANTITY * $TRAIN_PERCENT / 200" | bc)  # half of the training data from each type of images
TEST_QUANTITY=$((DATA_QUANTITY / 2 - TRAIN_QUANTITY))

# Training set
for i in $(seq 1 $TRAIN_QUANTITY) # {1..4750}
do
   cp "$ALGO_XX/$i.npz" "$TRAINING_IMAGES/$i.npz"
   cp "$MASKS_ALGO_XX/$i.npz" "$TRAINING_MASKS/$i.npz"
done

for i in $(seq $((TRAIN_QUANTITY + 1)) $((2 * TRAIN_QUANTITY))) # {4751..9500}
do
   cp "$NORMAL/$i.npz" "$TRAINING_IMAGES/$i.npz"
   cp "$MASKS_EMPTY/$i.npz" "$TRAINING_MASKS/$i.npz"
done

# Test set
for i in $(seq $((2 * TRAIN_QUANTITY + 1)) $((2 * TRAIN_QUANTITY + TEST_QUANTITY))) # {9501..9750}
do
   cp "$ALGO_XX/$i.npz" "$TEST_IMAGES/$i.npz"
   cp "$MASKS_ALGO_XX/$i.npz" "$TEST_MASKS/$i.npz"
done

for i in $(seq $((2 * TRAIN_QUANTITY + TEST_QUANTITY + 1)) $((2 * (TRAIN_QUANTITY + TEST_QUANTITY)))) # {9751..10000}
do
   cp "$NORMAL/$i.npz" "$TEST_IMAGES/$i.npz"
   cp "$MASKS_EMPTY/$i.npz" "$TEST_MASKS/$i.npz"
done
