# Copyright (c) 2018 NVIDIA Corporation
import time

import numpy as np
import tensorflow as tf

from open_seq2seq.utils.utils import get_base_config, check_logdir,\
                                     create_model, deco_print
from open_seq2seq.models.text2speech import save_audio

if __name__ == '__main__':
  # Define the command line arguments that one would pass to run.py here
  config_file_path = "example_configs/text2speech/tacotron_gst.py"
  checkpoint_path = "result/tacotron-gst-8gpu/logs/"
  syn_save_dir = "/data/speech/LibriSpeech-Syn/syn"

  args_T2S = ["--config_file={}".format(config_file_path),
              "--mode=infer",
              "--logdir={}".format(checkpoint_path),
              "--batch_size_per_gpu=256",
              "--infer_output_file=",
              "--num_gpus=1",
              "--use_horovod=False"]

  # A simpler version of what run.py does. It returns the created model and
  # its saved checkpoint
  def get_model(args):
    args, base_config, base_model, config_module = get_base_config(args)
    checkpoint = check_logdir(args, base_config)
    model = create_model(args, base_config, config_module, base_model, None)
    return model, checkpoint

  # A variant of iterate_data
  def iterate_data(model, sess, verbose, num_steps=None):
    # Helper function to save audio
    def infer(outputs, i):
      predicted_final_specs = outputs[1]
      sequence_lengths = outputs[4]
      for j in range(len(predicted_final_specs)):
        predicted_final_spec = predicted_final_specs[j]
        audio_length = sequence_lengths[j]

        if audio_length > 2:
          if "both" in model.get_data_layer().params['output_type']:
            predicted_mag_spec = outputs[5][j][:audio_length - 1, :]
          else:
            predicted_final_spec = predicted_final_spec[:audio_length - 1, :]
            predicted_mag_spec = model.get_data_layer().get_magnitude_spec(
                predicted_final_spec, is_mel=True)
          save_audio(
              predicted_mag_spec,
              syn_save_dir,
              0,
              n_fft=model.get_data_layer().n_fft,
              sampling_rate=model.get_data_layer().sampling_rate,
              mode="syn",
              number=i * batch_size + j,
              save_format="disk",
              gl_iters=4,
              verbose=False
          )
        else:
          print("WARNING: An audio file was not saved, this will error out in"
                "future steps")

    total_time = 0.0
    bench_start = model.params.get('bench_start', 10)

    size_defined = model.get_data_layer().get_size_in_samples() is not None
    if size_defined:
      dl_sizes = []

    total_samples = []
    fetches = []

    # on horovod num_gpus is 1
    for worker_id in range(model.num_gpus):
      cur_fetches = [
          model.get_data_layer(worker_id).input_tensors,
          model.get_output_tensors(worker_id),
      ]
      if size_defined:
        dl_sizes.append(model.get_data_layer(worker_id).get_size_in_samples())
      try:
        total_objects = 0.0
        cur_fetches.append(model.get_num_objects_per_step(worker_id))
      except NotImplementedError:
        total_objects = None
        deco_print("WARNING: Can't compute number of objects per step, since "
                   "train model does not define get_num_objects_per_step method.")
      fetches.append(cur_fetches)
      total_samples.append(0.0)

    sess.run([model.get_data_layer(i).iterator.initializer
              for i in range(model.num_gpus)])

    step = 0
    processed_batches = 0
    if verbose:
      if model.on_horovod:
        ending = " on worker {}".format(model.hvd.rank())
      else:
        ending = ""

    while True:
      tm = time.time()
      fetches_vals = {}
      if size_defined:
        fetches_to_run = {}
        # removing finished data layers
        for worker_id in range(model.num_gpus):
          if total_samples[worker_id] < dl_sizes[worker_id]:
            fetches_to_run[worker_id] = fetches[worker_id]
        fetches_vals = sess.run(fetches_to_run)
      else:
        # if size is not defined we have to process fetches sequentially, so not
        # to lose data when exception is thrown on one data layer
        for worker_id, one_fetch in enumerate(fetches):
          try:
            fetches_vals[worker_id] = sess.run(one_fetch)
          except tf.errors.OutOfRangeError:
            continue

      if step >= bench_start:
        total_time += time.time() - tm

      # looping over num_gpus. In Horovod case this loop is "dummy",
      # since num_gpus = 1
      for worker_id, fetches_val in fetches_vals.items():
        inputs, outputs = fetches_val[:2]

        if total_objects is not None:
          total_objects += np.sum(fetches_val[-1])

        # assuming any element of inputs["source_tensors"] .shape[0] is batch size
        batch_size = inputs["source_tensors"][0].shape[0]
        total_samples[worker_id] += batch_size

        if size_defined:
          # this data_layer is at the last batch with few more elements, cutting
          if total_samples[worker_id] > dl_sizes[worker_id]:
            last_batch_size = dl_sizes[worker_id] % batch_size
            for key, value in inputs.items():
              inputs[key] = model.clip_last_batch(value, last_batch_size)
            outputs = model.clip_last_batch(outputs, last_batch_size)

        infer(outputs, processed_batches)

        processed_batches += 1


      if verbose:
        if size_defined:
          data_size = int(np.sum(np.ceil(np.array(dl_sizes) / batch_size)))
          if step == 0 or len(fetches_vals) == 0 or \
             (data_size > 10 and processed_batches % (data_size // 10) == 0):
            deco_print("Processed {}/{} batches{}".format(
                processed_batches, data_size, ending
            ))
        else:
          deco_print("Processed {} batches{}".format(processed_batches, ending),
                     end='\r')

      if len(fetches_vals) == 0:
        break
      step += 1
      # break early in the case of INT8 calibration
      if num_steps is not None and step >= num_steps:
        break

    if verbose:
      if step > bench_start:
        deco_print(
            "Avg time per step{}: {:.3}s".format(
                ending, 1.0 * total_time / (step - bench_start)
            ),
        )
        if total_objects is not None:
          avg_objects = 1.0 * total_objects / total_time
          deco_print("Avg objects per second{}: {:.3f}".format(ending,
                                                               avg_objects))
      else:
        deco_print("Not enough steps for benchmarking{}".format(ending))

  model_T2S, checkpoint_T2S = get_model(args_T2S)

  # Create the session and load the checkpoints
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess = tf.InteractiveSession(config=sess_config)
  saver_T2S = tf.train.Saver()
  saver_T2S.restore(sess, checkpoint_T2S)

  iterate_data(model_T2S, sess, True)
