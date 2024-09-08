# /home/nicolasu/Documents/Diploma\ Picture\ sets/WOW/executable/WOW
# /home/nicolasu/Documents/Diploma\ Picture\ sets/HUGO/executable/HUGO_like

#/home/nicolasu/Documents/Diploma\ Picture\ sets/WOW/executable/WOW -v -a 0.25 -I ./NORMAL_ALL -O ./WOW_25_ALL
#/home/nicolasu/Documents/Diploma\ Picture\ sets/HUGO/executable/HUGO_like -v -a 0.25 -I ./NORMAL_ALL -O ./HUGO_25_ALL

# /home/nicolasu/Documents/Diploma\ Picture\ sets/WOW/executable/WOW -v -a 0.1 -I ./NORMAL_ALL -O ./WOW_10_ALL
# /home/nicolasu/Documents/Diploma\ Picture\ sets/HUGO/executable/HUGO_like -v -a 0.1 -I ./NORMAL_ALL -O ./HUGO_10_ALL

./mask_dir ./NORMAL_ALL ./HUGO_10_ALL ./MASK-HUGO_10_ALL
./mask_dir ./NORMAL_ALL ./HUGO_25_ALL ./MASK-HUGO_25_ALL

./mask_dir ./NORMAL_ALL ./WOW_10_ALL ./MASK-WOW_10_ALL
./mask_dir ./NORMAL_ALL ./WOW_25_ALL ./MASK-WOW_25_ALL

