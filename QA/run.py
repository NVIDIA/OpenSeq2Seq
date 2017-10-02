import json
import copy
import time
import os
import pprint
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.core.framework import summary_pb2
from rNet.rNet_model import BasicRNet
from rNet.rNet_baseline import BasicRNet_baseline
from rNet.data_layer import ParallelDataInRamInputLayer
from eval_SQuAD import f1_score, exact_match_score, metric_max_over_ground_truths

tf.flags.DEFINE_string("config_file", "",
						"""Path to the file with configuration""")
tf.flags.DEFINE_string("logdir", "",
						"""Path to where save logs and checkpoints""")
tf.flags.DEFINE_string("inference_out", "",
						"""where to output inference results""")
tf.flags.DEFINE_integer("checkpoint_frequency", 300,
						"""iterations after which a checkpoint is made. Only the last 100 checkpoints are saved""")
tf.flags.DEFINE_integer("summary_frequency", 100,
						"""iterations after which validation takes place""")

FLAGS = tf.flags.FLAGS

def train(config):
	deco_print('Executing training mode')
	deco_print('Creating data layer')
	dl = ParallelDataInRamInputLayer(params=config)

	do_eval = False
	if 'source_file_eval' in config and 'target_file_eval' in config:
		do_eval = True
		eval_config = copy.deepcopy(config)
		eval_config['source_file'] = eval_config['source_file_eval']
		eval_config['target_file'] = eval_config['target_file_eval']
		eval_config['target_candidate_file'] = eval_config['target_candidate_file_eval'] if 'target_candidate_file_eval' in eval_config else None
		deco_print('Creating eval data layer')
		eval_dl = ParallelDataInRamInputLayer(params=eval_config)

	deco_print('Data layer created')
	with tf.Graph().as_default():
		global_step = tf.contrib.framework.get_or_create_global_step()
		if 'baseline' in config and config['baseline'] == True:
			model = BasicRNet_baseline(model_params=config, global_step=global_step, embedding=dl.embedding)
		else:
			model = BasicRNet(model_params=config, global_step=global_step, embedding=dl.embedding)

		tf.summary.scalar(name='loss', tensor=model.loss)
		if do_eval:
			eval_fetches = [model.loss, model.predict]
		summary_op = tf.summary.merge_all()
		fetches_s = [model.loss, model.train_op, model.predict]
		sess_config = tf.ConfigProto(allow_soft_placement=True)
		saver = tf.train.Saver()
		epoch_saver = tf.train.Saver(max_to_keep=100)

		with tf.Session(config=sess_config) as sess:
			sw = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
			if tf.train.latest_checkpoint(FLAGS.logdir) is not None:
				saver.restore(sess, tf.train.latest_checkpoint(FLAGS.logdir))
				deco_print('Restored checkpoint. Resuming training')
			else:
				sess.run(tf.global_variables_initializer())

			for epoch in range(config['num_epochs']):
				deco_print('\n\n')
				deco_print('Doing epoch {}'.format(epoch))
				epoch_start = time.time()
				total_train_loss = 0.0

				total_train_position_accuracy = 0.0
				total_train_EM_accuracy = 0.0
				total_train_F1_accuracy = 0.0
				
				t_cnt = 0

				for i, (x, x_char, y, y_char, z, z_pos, bucket_id, len_x, len_x_char, len_y, len_y_char, len_z, z_cand) in enumerate(dl.iterate_one_epoch()):
					if i % FLAGS.checkpoint_frequency == 0 and i > 0:
						deco_print('Saving checkpoint')
						saver.save(sess, save_path=os.path.join(FLAGS.logdir, 'model'))

					loss, _, pos = sess.run(fetches=fetches_s, feed_dict={
						model.x: x,
						model.x_char: x_char,
						model.y: y,
						model.y_char: y_char,
						model.x_length: len_x,
						model.x_char_length: len_x_char,
						model.y_length: len_y,
						model.y_char_length: len_y_char,
						model.z_pos: z_pos})

					if i % FLAGS.summary_frequency == 0:
						sm, = sess.run(fetches=[summary_op], feed_dict={
							model.x: x,
							model.x_char: x_char,
							model.y: y,
							model.y_char: y_char,
							model.x_length: len_x,
							model.x_char_length: len_x_char,
							model.y_length: len_y,
							model.y_char_length: len_y_char,
							model.z_pos: z_pos})

						sw.add_summary(sm, global_step=sess.run(global_step))
						deco_print('In epoch {}, step {} the loss is {}'.format(epoch, i, loss))
						deco_print("TRAIN CONTEXT[0]: " + dl.pretty_print_array(x[0,:],
							vocab=dl.idx2seq,
							delim=config["delimiter"]))
						deco_print("TRAIN QUESTION[0]: " + dl.pretty_print_array(y[0,:],
							vocab=dl.idx2seq,
							delim = config["delimiter"]))
						deco_print("TRAIN ANSWER[0]: " + dl.pretty_print_array(z[0,:],
							vocab=dl.idx2seq,
							delim = config["delimiter"]))
						deco_print("TRAIN PREDICTION[0]: " + dl.pretty_print_array(x[0,pos[0,0]:pos[0,1]],
							vocab=dl.idx2seq,
							delim=config["delimiter"]))

					total_train_position_accuracy += 1.0 * np.sum(z_pos == pos) / pos.shape[0] / 2
					pred = [x[i,pos[i,0]:pos[i,1]] for i in range(pos.shape[0])]

					epoch_EM_accuracy = 0.0
					epoch_F1_accuracy = 0.0
					for pred_i, cand_i in zip(pred, z_cand):
						hyp = dl.pretty_print_array(pred_i, vocab=dl.idx2seq, ignore_special=True, delim=config["delimiter"])
						ref = [dl.pretty_print_array(ground_truth, vocab=dl.idx2seq, ignore_special=True, delim=config["delimiter"]) for ground_truth in cand_i]
						epoch_EM_accuracy += metric_max_over_ground_truths(exact_match_score, hyp, ref)
						epoch_F1_accuracy += metric_max_over_ground_truths(f1_score, hyp, ref)
					total_train_EM_accuracy += epoch_EM_accuracy / pos.shape[0]
					total_train_F1_accuracy += epoch_F1_accuracy / pos.shape[0]

					total_train_loss += loss
					t_cnt += 1

				epoch_end = time.time()
				deco_print('EPOCH {} TRAINING LOSS: {}'.format(epoch, total_train_loss / t_cnt))
				deco_print('EPOCH {} TRAINING POSITION ACCURACY: {}'.format(epoch, total_train_position_accuracy / t_cnt))
				deco_print('EPOCH {} TRAINING EM ACCURACY: {}'.format(epoch, total_train_EM_accuracy / t_cnt))
				deco_print('EPOCH {} TRAINING F1 ACCURACY: {}'.format(epoch, total_train_F1_accuracy / t_cnt))

				value_loss = summary_pb2.Summary.Value(tag="TrainEpochLoss", simple_value=total_train_loss / t_cnt)
				value_position_accuracy = summary_pb2.Summary.Value(tag="TrainEpochAccuracy", simple_value=total_train_position_accuracy / t_cnt)
				value_EM_accuracy = summary_pb2.Summary.Value(tag="TrainEpochEMAccuracy", simple_value=total_train_EM_accuracy / t_cnt)
				value_F1_accuracy = summary_pb2.Summary.Value(tag="TrainEpochF1Accuracy", simple_value=total_train_F1_accuracy / t_cnt)
				summary = summary_pb2.Summary(value=[value_loss, value_position_accuracy, value_EM_accuracy, value_F1_accuracy])
				sw.add_summary(summary=summary, global_step=sess.run(global_step))
				sw.flush()
				deco_print("Did epoch {} in {} seconds".format(epoch, epoch_end - epoch_start))
				dl.bucketize()

				if do_eval:
					deco_print("Evaluation on validation set")
					total_eval_loss = 0.0

					total_eval_position_accuracy = 0.0
					total_eval_EM_accuracy = 0.0
					total_eval_F1_accuracy = 0.0

					cnt = 0
					for i, (x, x_char, y, y_char, z, z_pos, bucket_id, len_x, len_x_char, len_y, len_y_char, len_z, z_cand) in enumerate(eval_dl.iterate_one_epoch()):
						e_loss, pos = sess.run(fetches=eval_fetches,feed_dict={
							model.x: x,
							model.x_char: x_char,
							model.y: y,
							model.y_char: y_char,
							model.x_length: len_x,
							model.x_char_length: len_x_char,
							model.y_length: len_y,
							model.y_char_length: len_y_char,
							model.z_pos: z_pos})

						total_eval_position_accuracy += 1.0 * np.sum(z_pos == pos) / pos.shape[0] / 2
						pred = [x[i,pos[i,0]:pos[i,1]] for i in range(pos.shape[0])]

						epoch_EM_accuracy = 0.0
						epoch_F1_accuracy = 0.0
						for pred_i, cand_i in zip(pred, z_cand):
							hyp = eval_dl.pretty_print_array(pred_i, vocab=dl.idx2seq, ignore_special=True, delim=config["delimiter"])
							ref = [eval_dl.pretty_print_array(ground_truth, vocab=dl.idx2seq, ignore_special=True, delim=config["delimiter"]) for ground_truth in cand_i]
							epoch_EM_accuracy += metric_max_over_ground_truths(exact_match_score, hyp, ref)
							epoch_F1_accuracy += metric_max_over_ground_truths(f1_score, hyp, ref)
						total_eval_EM_accuracy += epoch_EM_accuracy / pos.shape[0]
						total_eval_F1_accuracy += epoch_F1_accuracy / pos.shape[0]
						
						total_eval_loss += e_loss
						cnt += 1

					value_loss = summary_pb2.Summary.Value(tag="EvalLoss", simple_value=total_eval_loss/cnt)
					value_position_accuracy = summary_pb2.Summary.Value(tag="EvalAccuracy", simple_value=total_eval_position_accuracy/cnt)
					value_EM_accuracy = summary_pb2.Summary.Value(tag="EvalEMAccuracy", simple_value=total_eval_EM_accuracy/cnt)
					value_F1_accuracy = summary_pb2.Summary.Value(tag="EvalF1Accuracy", simple_value=total_eval_F1_accuracy/cnt)
					summary = summary_pb2.Summary(value=[value_loss, value_position_accuracy, value_EM_accuracy, value_F1_accuracy])
					sw.add_summary(summary=summary, global_step=sess.run(global_step))
					sw.flush()
					deco_print('EPOCH {} EVALUATION LOSS: {}'.format(epoch, total_eval_loss/cnt))
					deco_print('EPOCH {} EVALUATION POSITION ACCURACY: {}'.format(epoch, total_eval_position_accuracy/cnt))
					deco_print('EPOCH {} EVALUATION EM ACCURACY: {}'.format(epoch, total_eval_EM_accuracy/cnt))
					deco_print('EPOCH {} EVALUATION F1 ACCURACY: {}'.format(epoch, total_eval_F1_accuracy/cnt))
				deco_print("Saving Epoch checkpoint")
				epoch_saver.save(sess, save_path=os.path.join(FLAGS.logdir, "model-epoch"), global_step=epoch)

			# end of epoch loop
			deco_print("Saving last checkpoint")
			saver.save(sess, save_path=os.path.join(FLAGS.logdir, "model"), global_step=global_step)

def infer(config):
	deco_print('Executing inference mode')
	deco_print('Creating data layer')
	dl = ParallelDataInRamInputLayer(params=config)
	deco_print('Data layer created')

	with tf.Graph().as_default():
		global_step = tf.contrib.framework.get_or_create_global_step()
		if 'baseline' in config and config['baseline'] == True:
			model = BasicRNet_baseline(model_params=config, global_step=global_step, embedding=dl.embedding)
		else:
			model = BasicRNet(model_params=config, global_step=global_step, embedding=dl.embedding)
		
		fetches = [model.predict]
		sess_config = tf.ConfigProto(allow_soft_placement=True)
		saver = tf.train.Saver()
		
		with tf.Session(config=sess_config) as sess:
			if tf.train.latest_checkpoint(FLAGS.logdir) is not None:
				saver.restore(sess, tf.train.latest_checkpoint(FLAGS.logdir))
				deco_print('Restored checkpoint.')
			else:
				sess.run(tf.global_variables_initializer())
				deco_print('Ramdon initialization.')
			deco_print('Saving inference results to: ' + FLAGS.inference_out)
			start = time.time()
			with open(FLAGS.inference_out, 'w') as f_out:
				for i, (x, x_char, y, y_char, z, _, bucket_id, len_x, len_x_char, len_y, len_y_char, _, _) in enumerate(dl.iterate_one_epoch()):
					ans, = sess.run(fetches=fetches,feed_dict={
						model.x: x,
						model.x_char: x_char,
						model.y: y,
						model.y_char: y_char,
						model.x_length: len_x,
						model.x_char_length: len_x_char,
						model.y_length: len_y,
						model.y_char_length: len_y_char})
					deco_print("TEST CONTEXT[0]: " + dl.pretty_print_array(x[0,:], vocab=dl.idx2seq, delim=config["delimiter"]))
					deco_print("TEST QUESTION[0]: " + dl.pretty_print_array(y[0,:], vocab=dl.idx2seq, delim = config["delimiter"]))
					deco_print("TEST ANSWER[0]: " + dl.pretty_print_array(z[0,:], vocab=dl.idx2seq, delim = config["delimiter"]))
					deco_print("TEST PREDICTION[0]: " + dl.pretty_print_array(x[0,ans[0,0]:ans[0,1]], vocab=dl.idx2seq, delim=config["delimiter"]))
					for ii in range(ans.shape[0]):
						f_out.write('%d\t%d\t%s\n' %(ans[ii,0], ans[ii,1], dl.pretty_print_array(x[ii, ans[ii,0]:ans[ii,1]], vocab=dl.idx2seq, ignore_special=True, delim=config["delimiter"])))
			deco_print("Inference finished in {} seconds\n\n".format(time.time() - start))

def deco_print(line):
	print(">==================> " + line)

def main(_):
	with open(FLAGS.config_file) as data_file:
		config = json.load(data_file)
	deco_print("TensorFlow version: " + tf.__version__)
	print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
	print("Read the following in config:")
	pprint.pprint(config)
	print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
	if config["mode"] == "train":
		deco_print("Running in training mode")
		train(config)
	elif config["mode"] == "infer":
		deco_print("Running in inference mode")
		infer(config)
	else:
		raise ValueError("Unknown mode in config file")

if __name__ == "__main__":
	tf.app.run()
