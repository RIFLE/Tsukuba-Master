# #!/bin/bash

# Set total images
#total_images=10000

# Check if each image exists
#for (( img=1; img<=$total_images; img++ ))
#do
    # If the file does not exist, print its name
#    if [[ ! -e "${img}.pgm" ]]; then
#        echo "${img}.pgm"
#    fi
#done

# #!/bin/bash

# Set total images
total_images=10000

# Check if each image exists in each batch
for (( batch=1; batch<=10; batch++ ))
do
    # Calculate the range of the batch
    start=$(( (batch - 1) * 1000 + 1 ))
    end=$(( batch * 1000 ))
    
    # Flag to check if the batch is complete
    batch_complete=true

    for (( img=$start; img<=$end; img++ ))
    do
        # If the file does not exist, mark the batch as incomplete
        if [[ ! -e "${img}.pgm" ]]; then
            batch_complete=false
            break
        fi
    done

    # If the batch is incomplete, print its number
    if [[ $batch_complete == false ]]; then
        echo "Batch $batch is incomplete"
    fi
done
