import os
PATH = 'data_45k/attributes_dataset_train_new_median.txt'


f = open(PATH,"r")
lines = f.readlines()

out = open('out.txt',"w")

head = lines[0].split()
index = head.index('glossy') + 1
print(index)
for line in lines[1:]:
    values = line.split()
    if float(values[index]) == 0.0:
        out.write(line)
        

out.close()

out = open('out.txt',"r")
lines = out.readlines()

for line in lines[1:]:
    values = line.split()
    os.system("cp data_45k/renderings/" + values[0][:len(line)-1] + " data_45k/renderings/no_glossy")
    
    
