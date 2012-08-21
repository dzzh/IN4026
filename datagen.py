import random

FILE_NAME = "1m.test"
NUM_ENTRIES = 1048576
LARGEST_NUM = 200
BORDER = 10

with open(FILE_NAME, mode="w") as a_file:
	for	i in range(BORDER):
		a_file.write(str(random.randrange(0,10)) + " ")
	for i in range(NUM_ENTRIES-BORDER):
		a_file.write(str(random.randrange(10,LARGEST_NUM)) + " ")
	for	i in range(BORDER):
		a_file.write(str(random.randrange(0,10)) + " ")
