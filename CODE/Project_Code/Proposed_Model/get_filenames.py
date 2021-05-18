import random

with open("/home/darshan/data/ug2challengedata/casia/lensless/casia_new_measurements_list.txt") as fp:
    file_names = fp.readlines()
    random.shuffle(file_names)
    print(type(file_names))
    print(len(file_names))
    file_names_train = file_names[:10000]
    file_names_test = file_names[10000:11000]

with open("casia_file_names_train.txt", "w") as fp1:
    for fn in file_names_train:
       fp1.write(fn.split(" ")[0] + "\n")
with open("casia_file_names_test.txt", "w") as fp2:
    for fn in file_names_test:
       fp2.write(fn.split(" ")[0] + "\n")
    
