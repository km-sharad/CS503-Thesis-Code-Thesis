import csv

'''
Filters training images with visible car door handle
'''

output = open('cdhd_anno_training_data.txt', 'w')
i = 1

with open("cdhd_anno.txt") as f:
	for row in iter(f):
		split_row = row.split('|')
		#row[4] = isTrain, row[6] = isVisible
		if int(split_row[4]) == 1 and int(split_row[6]) == 1:
			output.write(row)
			i = i + 1
print i
output.close()