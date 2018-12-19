# Produces labelled .csv files containing balanced samples of classes
# and their labels for the chosen dataset
import os
import random
import librosa
import numpy as np


# choose one of three datasets
# 	1) v1-12: V1 dataset with 12 classes, including unknown and silence
# 	2) v1-30: V1 dataset with 30 classes, without unknown and silence
# 	3) v2: V2 dataset with 35 classes
DATASET = "v1-12"

if DATASET == "v1-12":
	classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]
elif DATASET == "v1-30":
	classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "bed", "bird", "cat", "dog", "eight", "five", "four", "happy", "house", "marvin", "nine", "one", "seven", "sheila", "six", "three", "tree", "two", "wow", "zero"]
elif DATASET == "v2":
	classes = ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"]
else:
	print("Dataset not supported")
	exit()

root_dir = "../../../data"
if "v1" in DATASET:
	root_dir = os.path.join(root_dir, "speech_commands_v0.01")
else:
	root_dir = os.path.join(root_dir, "speech_commands_v0.02")


eval_batch = 16
train_split = 0.8
test_split = val_split = (1 - train_split) / 2

data_list = []
min_samples_per_class = None
max_samples_per_class = 5000

# build a list of all available samples
for idx, label in enumerate(classes):
	class_list = []

	if label == "unknown":
		unknowns = ["bed", "bird", "cat", "dog", "eight", "five", "four", "happy", "house", "marvin", "nine", "one", "seven", "sheila", "six", "three", "tree", "two", "wow", "zero"]
		
		for unknown in unknowns:
			folder = os.path.join(root_dir, unknown)

			for file in os.listdir(folder):
				file_path = "{}/{}".format(unknown, file)
				class_list.append(file_path)

	elif label == "silence":
		silence_path = os.path.join(root_dir, "silence")
		if not os.path.exists(silence_path):
			os.mkdir(silence_path)

		silence_stride = 2000
		sampling_rate = 16000
		folder = os.path.join(root_dir, "_background_noise_")

		for file in os.listdir(folder):
			if ".wav" in file:
				load_path = os.path.join(folder, file)
				y, sr = librosa.load(load_path)

				for i in range(0, len(y) - sampling_rate, silence_stride):
					file_path = "silence/{}_{}.wav".format(file[:-4], i)
					y_slice = y[i:i + sampling_rate]
					librosa.output.write_wav(os.path.join(root_dir, file_path), y_slice, sr)
					class_list.append(file_path)

	else:
		folder = os.path.join(root_dir, label)

		for file in os.listdir(folder):
			file_path = "{}/{}".format(label, file)
			class_list.append(file_path)

	if min_samples_per_class is None or len(class_list) < min_samples_per_class:
		min_samples_per_class = len(class_list)

	random.shuffle(class_list)
	data_list.append(class_list)


# sample and write to files
test_part = int(test_split * min_samples_per_class)
test_part += eval_batch - (test_part % eval_batch)
val_part = int(test_split * min_samples_per_class)
val_part += eval_batch - (val_part % eval_batch)

train_samples = []
test_samples = []
val_samples = []

for i, class_list in enumerate(data_list):
	# take test and validation samples out
	for sample in class_list[:test_part]:
		test_samples.append("{},{}".format(i, sample))
	for sample in class_list[test_part:test_part + val_part]:
		val_samples.append("{},{}".format(i, sample))

	samples = class_list[test_part + val_part:]
	length = len(samples)

	while len(class_list) < max_samples_per_class:
		l = np.random.randint(0, length)
		class_list.append(samples[l])

	for sample in class_list[test_part + val_part:max_samples_per_class]:
		train_samples.append("{},{}".format(i, sample))

train_file = open(os.path.join(root_dir, DATASET + "-train.txt"), "w")
for line in train_samples:
	train_file.write(line + "\n")
test_file = open(os.path.join(root_dir, DATASET + "-test.txt"), "w")
for line in test_samples:
	test_file.write(line + "\n")
val_file = open(os.path.join(root_dir, DATASET + "-val.txt"), "w")
for line in val_samples:
	val_file.write(line + "\n")
	