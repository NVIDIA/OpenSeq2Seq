output_file = 'outputs/sst-wk103-exp4.txt'
# output_file = 'outputs/imdb-wkt2-exp6.txt'
# output_file = 'outputs/imdb-wkt103-exp1.txt'

output = open(output_file, 'r')

keyword = "***     EVAL Accuracy: "

best_acc = 0.0
# pre, rec, f1, loss = 0.0, 0.0, 0.0, 0.0 
loss = ''
stat, step = '', ''
get_stat = False
n = len(keyword)
m = len("***     Validation loss: ")
last = ''
get_step = False
for line in output.readlines():
	line = line.strip()
	if get_stat:
		stat = line
		get_stat = False
		get_step = True
	elif get_step:
		step = line
		get_step = False
	else:
		idx = line.find(keyword)
		if idx != -1:
			acc = float(line[n:])
			if acc > best_acc:
				best_acc = acc
				loss = last
				get_stat = True
		last = line


print("***     Best accuracy:", str(best_acc))
print(loss)
print(stat)
print(step)