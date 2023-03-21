import os

file = open('/media/raid/dsubias/data_45k/attributes_dataset_real.txt')

Lines = file.readlines()
 
count = 0
# Strips the newline character
for line in Lines[1:]:
    
    imag_path = line.split()[0]

    os.system("cp {} ./test_images/".format('/media/raid/dsubias/data_45k/renderings/' + imag_path))
