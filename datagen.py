import random

FILE_NAME = "data.test"
NUM_ENTRIES = 262144
LARGEST_NUM = 200

with open(FILE_NAME, mode="w") as a_file:
	for i in range(NUM_ENTRIES):
		a_file.write(str(random.randrange(0,LARGEST_NUM)) + " ")