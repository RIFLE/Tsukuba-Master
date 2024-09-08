# ALGO_XX="./10000/WOW-10/conforming_10000/10000_WOW-10"
# MASKS_ALGO_XX="./10000/HUGO-25/conforming_10000/10000_MASKS-HUGO-25"

# NORMAL="./10000/10000_NORMAL_conforming"
# MASKS_EMPTY="./10000/10000_EMPTY_conforming"

# TRAINING_IMAGES="./HUGO_25_10000/training/images"
# TRAINING_MASKS="./HUGO_25_10000/training/masks"

# TEST_IMAGES="./HUGO_25_10000/testing/images"
# TEST_MASKS="./HUGO_25_10000/testing/masks"

# rsync -a -f"+ */" -f"- *" ./HUGO_25_10000/ ./WOW_10_10000/

cd ./HUGO_25_10000/testing/; for i in {9501..10000}; do cp ../../../../10000/HUGO-25/conforming_10000/10000_MASKS-HUGO-25/${i}.npz ../testing/masks/${i}.npz; done

echo "Testing masks done";

for i in {9501..10000}; do cp ../../../../10000/HUGO-25/conforming_10000/10000_HUGO-25/${i}.npz ../testing/images/${i}.npz; done

echo "Testing img done";

for i in {1..9500}; do cp ../../../../10000/HUGO-25/conforming_10000/10000_HUGO-25/${i}.npz ../training/images/${i}.npz; done

echo "Training img done";

for i in {1..9500}; do cp ../../../../10000/HUGO-25/conforming_10000/10000_MASKS-HUGO-25/${i}.npz ../training/masks/${i}.npz; done

echo "Training masks done";