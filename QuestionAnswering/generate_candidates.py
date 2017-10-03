import argparse
import os
import json

parser = argparse.ArgumentParser(description='Generate standard candidates file. ')
parser.add_argument('--source', help='path to the source file')
parser.add_argument('--file_id', help='path to id')
parser.add_argument('--target', help='path to the inference file')
### skipped data
parser.add_argument('--source_skip', help='path to the skipped source file')
parser.add_argument('--file_id_skip', help='path to skipped id')
parser.add_argument('--target_skip', help='path to the skipped inference file')
###
args = parser.parse_args()

path, filename = args.target.rsplit('/', 1)

ans_dict = {}
f = open(os.path.join(path, 'candidates.json'), 'w')
f_id = open(args.file_id, 'r')
f_tgt = open(args.target, 'r')
if args.source is not None:
	f_src = open(args.source, 'r')
for ans in f_tgt:
	if args.source is None:
		ans = ans.rstrip('\n').split('\t')[-1]
	else:
		start, end = ans.rstrip('\n').split('\t')[:-1]
		start = int(start)
		end = int(end)
		ctx_list = f_src.readline().rstrip('\n').split('<Q>')[0].split()
		if start >= end:
			ans = ''
		else:
			ans = ' '.join(ctx_list[start:end])
	ques_id = f_id.readline().rstrip('\n')
	ans_dict[ques_id] = ans
f_tgt.close()
f_id.close()
if args.source is not None:
	f_src.close()

if args.target_skip is not None and args.file_id_skip is not None:
	f_id = open(args.file_id_skip, 'r')
	f_tgt = open(args.target_skip, 'r')
	if args.source_skip is not None:
		f_src = open(args.source_skip, 'r')
	for ans in f_tgt:
		if args.source_skip is None:
			ans = ans.rstrip('\n').split('\t')[-1]
		else:
			start, end = ans.rstrip('\n').split('\t')[:-1]
			start = int(start)
			end = int(end)
			ctx_list = f_src.readline().rstrip('\n').split('<Q>')[0].split()
			if start >= end:
				ans = ''
			else:
				ans = ' '.join(ctx_list[start:end])
		ques_id = f_id.readline().rstrip('\n')
		ans_dict[ques_id] = ans
	f_tgt.close()
	f_id.close()
	if args.source_skip is not None:
		f_src.close()
json.dump(ans_dict, f)
f.close()