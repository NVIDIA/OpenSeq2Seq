import os, argparse, json, nltk

parser = argparse.ArgumentParser(description='Pre-process dataset.')
parser.add_argument('--file', help='path to the source file')
parser.add_argument('--mode', help='train/dev')
args = parser.parse_args()

path, filename = args.file.rsplit('/',1)
path_src = os.path.join(path, args.mode, 'src')
path_tgt = os.path.join(path, args.mode, 'tgt')

if not os.path.exists(path_src):
	os.makedirs(path_src)
if not os.path.exists(path_tgt):
	os.makedirs(path_tgt)

with open(args.file, 'r') as f:
	data = json.load(f)['data']

f_src = open(os.path.join(path_src, 'sources.txt'), 'w')
f_tgt = open(os.path.join(path_tgt, 'targets.txt'), 'w')
f_id = open(os.path.join(path, args.mode, 'id.txt'), 'w')
f_text = open(os.path.join(path, 'text.txt'), 'a')
if args.mode == 'dev':
	f_tgt_3 = open(os.path.join(path_tgt, 'targets_3.txt'), 'w')
	f_src_skip = open(os.path.join(path_src, 'sources_skip.txt'), 'w')
	f_tgt_skip = open(os.path.join(path_tgt, 'targets_skip.txt'), 'w')
	f_id_skip = open(os.path.join(path, args.mode, 'id_skip.txt'), 'w')

for article in data:
	title = article['title']
	paragraphs = article['paragraphs']
	for paragraph in paragraphs:
		context = paragraph['context']
		context_list = nltk.word_tokenize(context)
		f_text.write(' '.join(context_list) + '\n')
		qas = paragraph['qas']
		for qa in qas:
			question = qa['question']
			ques_id = qa['id']
			question_list = nltk.word_tokenize(question)
			answer_len_min = float('inf')
			answer_lists = []
			for i in range(len(qa['answers'])):
				answer_cand = qa['answers'][i]['text']
				answer_cand_list = nltk.word_tokenize(answer_cand)
				answer_lists.append(answer_cand_list)
				answer_cand_start = qa['answers'][i]['answer_start']
				if len(answer_cand_list) < answer_len_min:
					answer_len_min = len(answer_cand_list)
					answer = answer_cand
					answer_start = answer_cand_start
					answer_list = answer_cand_list

			# find answer positions
			idx = 0
			num_char = 0
			while idx < len(context_list) and num_char + len(context_list[idx]) < answer_start:
				num_char += len(context_list[idx])
				idx += 1
			idx = min(idx, len(context_list)-1)
			find = False
			while not find and idx >= 0:
				if idx + len(answer_list) > len(context_list):
					idx -= 1
					continue
				find = True
				for i in range(len(answer_list)):
					if context_list[idx+i] != answer_list[i]:
						find = False
						break
				if not find:
					idx -= 1

			if idx < 0:
				print('WARNING: skip question ==> ' + question)
				if args.mode == 'dev':
					source = ' '.join(['<P>']+context_list+['<Q>']+question_list)
					target = ' '.join(answer_list)
					f_src_skip.write(source+'\n')
					f_tgt_skip.write(target+'\t0\t1\n')
					f_id_skip.write(ques_id+'\n')
			else:
				pos_start = idx
				pos_end = idx + len(answer_list)
				source = ' '.join(['<P>']+context_list+['<Q>']+question_list)
				target = ' '.join(answer_list)
				f_src.write(source+'\n')
				f_tgt.write(target+'\t%d\t%d\n'%(pos_start+1,pos_end+1))
				f_id.write(ques_id+'\n')
				if args.mode == 'dev':
					target_3 = [' '.join(x) for x in answer_lists]
					f_tgt_3.write('\t'.join(target_3)+'\n')

f_src.close()
f_tgt.close()
f_text.close()
f_id.close()
if args.mode == 'dev':
	f_tgt_3.close()
	f_src_skip.close()
	f_tgt_skip.close()
	f_id_skip.close()
