'''
This file modifies standard TensorFlow modules necessary for transfer learning,
such as MonitoredTrainingSession, ChiefSessionCreator, Scaffold, SessionManager
'''
import re
import time

import tensorflow as tf
from tensorflow.python.ops import resources
from tensorflow.python.training import saver as training_saver

FP32_TEST = re.compile(r'Loss_Optimization\/FP32-master-copy\/')

# Value that indicates no value was provided.
USE_DEFAULT = object()

def TransferMonitoredTrainingSession(master='',  # pylint: disable=invalid-name
                                     is_chief=True,
                                     checkpoint_dir=None,
                                     scaffold=None,
                                     hooks=None,
                                     chief_only_hooks=None,
                                     save_checkpoint_secs=USE_DEFAULT,
                                     save_summaries_steps=USE_DEFAULT,
                                     save_summaries_secs=USE_DEFAULT,
                                     config=None,
                                     stop_grace_period_secs=120,
                                     log_step_count_steps=100,
                                     max_wait_secs=7200,
                                     save_checkpoint_steps=USE_DEFAULT,
                                     summary_dir=None,
                                     load_model_dir=None,
                                     load_fc=False):
  """Creates a `MonitoredSession` for training.
  For a chief, this utility sets proper session initializer/restorer. It also
  creates hooks related to checkpoint and summary saving. For workers, this
  utility sets proper session creator which waits for the chief to
  initialize/restore. Please check `tf.train.MonitoredSession` for more
  information.
  Args:
    master: `String` the TensorFlow master to use.
    is_chief: If `True`, it will take care of initialization and recovery the
      underlying TensorFlow session. If `False`, it will wait on a chief to
      initialize or recover the TensorFlow session.
    checkpoint_dir: A string.  Optional path to a directory where to restore
      variables.
    scaffold: A `Scaffold` used for gathering or building supportive ops. If
      not specified, a default one is created. It's used to finalize the graph.
    hooks: Optional list of `SessionRunHook` objects.
    chief_only_hooks: list of `SessionRunHook` objects. Activate these hooks if
      `is_chief==True`, ignore otherwise.
    save_checkpoint_secs: The frequency, in seconds, that a checkpoint is saved
      using a default checkpoint saver. If both `save_checkpoint_steps` and
      `save_checkpoint_secs` are set to `None`, then the default checkpoint
      saver isn't used. If both are provided, then only `save_checkpoint_secs`
      is used. Default 600.
    save_summaries_steps: The frequency, in number of global steps, that the
      summaries are written to disk using a default summary saver. If both
      `save_summaries_steps` and `save_summaries_secs` are set to `None`, then
      the default summary saver isn't used. Default 100.
    save_summaries_secs: The frequency, in secs, that the summaries are written
      to disk using a default summary saver.  If both `save_summaries_steps` and
      `save_summaries_secs` are set to `None`, then the default summary saver
      isn't used. Default not enabled.
    config: an instance of `tf.ConfigProto` proto used to configure the session.
      It's the `config` argument of constructor of `tf.Session`.
    stop_grace_period_secs: Number of seconds given to threads to stop after
      `close()` has been called.
    log_step_count_steps: The frequency, in number of global steps, that the
      global step/sec is logged.
    max_wait_secs: Maximum time workers should wait for the session to
      become available. This should be kept relatively short to help detect
      incorrect code, but sometimes may need to be increased if the chief takes
      a while to start up.
    save_checkpoint_steps: The frequency, in number of global steps, that a
      checkpoint is saved using a default checkpoint saver. If both
      `save_checkpoint_steps` and `save_checkpoint_secs` are set to `None`, then
      the default checkpoint saver isn't used. If both are provided, then only
      `save_checkpoint_secs` is used. Default not enabled.
    summary_dir: A string.  Optional path to a directory where to
      save summaries. If None, checkpoint_dir is used instead.
    load_model_dir (str): The location of the checkpoint file used to load the
      model weights.
  Returns:
    A `MonitoredSession` object.
  """
  if save_summaries_steps == USE_DEFAULT and save_summaries_secs == USE_DEFAULT:
    save_summaries_steps = 100
    save_summaries_secs = None
  elif save_summaries_secs == USE_DEFAULT:
    save_summaries_secs = None
  elif save_summaries_steps == USE_DEFAULT:
    save_summaries_steps = None

  if (save_checkpoint_steps == USE_DEFAULT and
      save_checkpoint_secs == USE_DEFAULT):
    save_checkpoint_steps = None
    save_checkpoint_secs = 600
  elif save_checkpoint_secs == USE_DEFAULT:
    save_checkpoint_secs = None
  elif save_checkpoint_steps == USE_DEFAULT:
    save_checkpoint_steps = None

  if not is_chief:
    session_creator = tf.train.WorkerSessionCreator(
        scaffold=scaffold,
        master=master,
        config=config,
        max_wait_secs=max_wait_secs)
    return tf.train.MonitoredSession(
        session_creator=session_creator, hooks=hooks or [],
        stop_grace_period_secs=stop_grace_period_secs)

  all_hooks = []
  if chief_only_hooks:
    all_hooks.extend(chief_only_hooks)

  restore_all = False
  if not load_model_dir:
    load_model_dir = checkpoint_dir
    restore_all = True

  assign_ops, restore_dict = get_assign_ops_and_restore_dict(
      tf.train.latest_checkpoint(load_model_dir), restore_all)

  if ((restore_all or tf.train.latest_checkpoint(checkpoint_dir))
      and len(assign_ops) == 0):
  # Checking to see if we can use the default TensorFlow Session Creator
  # We need two conditions to be true:
  # 1a) We are not loading partial vars through load_model_dir OR
  # 1b) There is a saved checkpoint file from which we can load
  # 2) if there is no dtype mismatch between checkpoint vars and vars in graph
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold,
        checkpoint_dir=checkpoint_dir,
        master=master,
        config=config)

  else: # load variables from the base model's checkpoint
    if load_model_dir:
      print("Loading the base model from {}.".format(load_model_dir))
    session_creator = TransferChiefSessionCreator(
        scaffold=scaffold,
        checkpoint_dir=load_model_dir,
        master=master,
        config=config,
        load_fc=load_fc,
        assign_ops=assign_ops,
        restore_dict=restore_dict)

  summary_dir = summary_dir or checkpoint_dir
  if summary_dir:
    if log_step_count_steps and log_step_count_steps > 0:
      all_hooks.append(
          tf.train.StepCounterHook(
              output_dir=summary_dir, every_n_steps=log_step_count_steps))

    if (save_summaries_steps and save_summaries_steps > 0) or (
        save_summaries_secs and save_summaries_secs > 0):
      all_hooks.append(tf.train.SummarySaverHook(
          scaffold=scaffold,
          save_steps=save_summaries_steps,
          save_secs=save_summaries_secs,
          output_dir=summary_dir))

  if checkpoint_dir:
    if (save_checkpoint_secs and save_checkpoint_secs > 0) or (
        save_checkpoint_steps and save_checkpoint_steps > 0):
      all_hooks.append(tf.train.CheckpointSaverHook(
          checkpoint_dir,
          save_steps=save_checkpoint_steps,
          save_secs=save_checkpoint_secs,
          scaffold=scaffold))

  if hooks:
    all_hooks.extend(hooks)
  return tf.train.MonitoredSession(
      session_creator=session_creator, hooks=all_hooks,
      stop_grace_period_secs=stop_grace_period_secs)

class TransferChiefSessionCreator(tf.train.SessionCreator):
  def __init__(self,
               scaffold=None,
               master='',
               config=None,
               checkpoint_dir=None,
               checkpoint_filename_with_path=None,
               load_fc=False,
               assign_ops=None,
               restore_dict=None):
    """Initializes a chief session creator.
    Args:
      scaffold: A `Scaffold` used for gathering or building supportive ops. If
        not specified a default one is created. It's used to finalize the graph.
      master: `String` representation of the TensorFlow master to use.
      config: `ConfigProto` proto used to configure the session.
      checkpoint_dir: A string.  Optional path to a directory where to restore
        variables.
      checkpoint_filename_with_path: Full file name path to the checkpoint file.
    """
    self._checkpoint_dir = checkpoint_dir
    self._checkpoint_filename_with_path = checkpoint_filename_with_path
    self._scaffold = scaffold or TransferScaffold()
    self._session_manager = None
    self._master = master
    self._config = config
    self._load_fc = load_fc
    self._assign_ops = assign_ops
    self._restore_dict = restore_dict

  def _get_session_manager(self):
    if self._session_manager:
      return self._session_manager

    self._session_manager = TransferSessionManager(
        local_init_op=self._scaffold.local_init_op,
        ready_op=self._scaffold.ready_op,
        ready_for_local_init_op=self._scaffold.ready_for_local_init_op,
        graph=tf.get_default_graph())
    return self._session_manager

  def create_session(self):
    print('SCAFFOLD TYPE:', type(self._scaffold))
    self._scaffold.finalize()
    # tf.get_default_graph()._unsafe_unfinalize()

    return self._get_session_manager().prepare_session(
        self._master,
        saver=self._scaffold.saver,
        checkpoint_dir=self._checkpoint_dir,
        checkpoint_filename_with_path=self._checkpoint_filename_with_path,
        config=self._config,
        init_op=self._scaffold.init_op,
        init_feed_dict=self._scaffold.init_feed_dict,
        init_fn=self._scaffold.init_fn,
        load_fc=self._load_fc,
        assign_ops=self._assign_ops,
        restore_dict=self._restore_dict)

class TransferScaffold(tf.train.Scaffold):
  def finalize(self):
    """Creates operations if needed and finalizes the graph."""
    if self._init_op is None:
      def default_init_op():
        return tf.group(
            tf.global_variables_initializer(),
            resources.initialize_resources(resources.shared_resources()))
      self._init_op = TransferScaffold.get_or_default(
          'init_op',
          tf.GraphKeys.INIT_OP,
          default_init_op)
    if self._ready_op is None:
      def default_ready_op():
        return tf.concat([
            tf.report_uninitialized_variables(),
            resources.report_uninitialized_resources()
        ], 0)
      self._ready_op = TransferScaffold.get_or_default(
          'ready_op', tf.GraphKeys.READY_OP,
          default_ready_op)
    if self._ready_for_local_init_op is None:
      def default_ready_for_local_init_op():
        return tf.report_uninitialized_variables(
            tf.global_variables())
      self._ready_for_local_init_op = TransferScaffold.get_or_default(
          'ready_for_local_init_op', tf.GraphKeys.READY_FOR_LOCAL_INIT_OP,
          default_ready_for_local_init_op)
    if self._local_init_op is None:
      self._local_init_op = TransferScaffold.get_or_default(
          'local_init_op', tf.GraphKeys.LOCAL_INIT_OP,
          TransferScaffold.default_local_init_op)
    if self._summary_op is None:
      self._summary_op = TransferScaffold.get_or_default(
          'summary_op', tf.GraphKeys.SUMMARY_OP, tf.summary.merge_all)
    # pylint: disable=g-long-lambda
    if self._saver is None:
      self._saver = training_saver._get_saver_or_default()  # pylint: disable=protected-access
    # pylint: enable=g-long-lambda
    self._saver.build()

    # ops.get_default_graph().finalize()
    # logging.info('Graph was finalized.')
    return self

class TransferSessionManager(tf.train.SessionManager):
  def _restore_checkpoint(self,
                          master,
                          sess,
                          saver=None,
                          checkpoint_dir=None,
                          checkpoint_filename_with_path=None,
                          wait_for_checkpoint=False,
                          max_wait_secs=7200,
                          config=None,
                          load_fc=False,
                          assign_ops=None,
                          restore_dict=None):
    """Creates a `Session`, and tries to restore a checkpoint.
    Args:
      master: `String` representation of the TensorFlow master to use.
      saver: A `Saver` object used to restore a model.
      checkpoint_dir: Path to the checkpoint files. The latest checkpoint in the
        dir will be used to restore.
      checkpoint_filename_with_path: Full file name path to the checkpoint file.
      wait_for_checkpoint: Whether to wait for checkpoint to become available.
      max_wait_secs: Maximum time to wait for checkpoints to become available.
      config: Optional `ConfigProto` proto used to configure the session.
    Returns:
      A pair (sess, is_restored) where 'is_restored' is `True` if
      the session could be restored, `False` otherwise.
    Raises:
      ValueError: If both checkpoint_dir and checkpoint_filename_with_path are
        set.
    """
    self._target = master
    # sess = tf.Session(self._target, graph=self._graph, config=config)

    if checkpoint_dir and checkpoint_filename_with_path:
      raise ValueError("Can not provide both checkpoint_dir and "
                       "checkpoint_filename_with_path.")
    # If either saver or checkpoint_* is not specified, cannot restore. Just
    # return.
    print('checkpoint_dir', checkpoint_dir)
    print('checkpoint_filename_with_path', checkpoint_filename_with_path)
    if not saver or not (checkpoint_dir or checkpoint_filename_with_path):
      return sess, False

    if checkpoint_filename_with_path:
      # saver.restore(sess, checkpoint_filename_with_path)
      # restore_certain_variables(sess, checkpoint_filename_with_path)
      run_assign_and_saver(
          sess, checkpoint_filename_with_path, assign_ops, restore_dict)
      return sess, True

    # Waits up until max_wait_secs for checkpoint to become available.
    wait_time = 0
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    while not ckpt or not ckpt.model_checkpoint_path:
      if wait_for_checkpoint and wait_time < max_wait_secs:
        tf.logging.info("Waiting for checkpoint to be available.")
        time.sleep(self._recovery_wait_secs)
        wait_time += self._recovery_wait_secs
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      else:
        return sess, False

    # Loads the checkpoint.
    ckpt_file = ckpt.model_checkpoint_path
    # restore_certain_variables(sess, ckpt_file)
    run_assign_and_saver(sess, ckpt_file, assign_ops, restore_dict)
    saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
    return sess, True

  def prepare_session(self,
                      master,
                      init_op=None,
                      saver=None,
                      checkpoint_dir=None,
                      checkpoint_filename_with_path=None,
                      wait_for_checkpoint=False,
                      max_wait_secs=7200,
                      config=None,
                      init_feed_dict=None,
                      init_fn=None,
                      load_fc=False,
                      assign_ops=None,
                      restore_dict=None):
    """Creates a `Session`. Makes sure the model is ready to be used.
      Creates a `Session` on 'master'. If a `saver` object is passed in, and
      `checkpoint_dir` points to a directory containing valid checkpoint
      files, then it will try to recover the model from checkpoint. If
      no checkpoint files are available, and `wait_for_checkpoint` is
      `True`, then the process would check every `recovery_wait_secs`,
      up to `max_wait_secs`, for recovery to succeed.
      If the model cannot be recovered successfully then it is initialized by
      running the `init_op` and calling `init_fn` if they are provided.
      The `local_init_op` is also run after init_op and init_fn, regardless of
      whether the model was recovered successfully, but only if
      `ready_for_local_init_op` passes.
      If the model is recovered from a checkpoint it is assumed that all
      global variables have been initialized, in particular neither `init_op`
      nor `init_fn` will be executed.
      It is an error if the model cannot be recovered and no `init_op`
      or `init_fn` or `local_init_op` are passed.
      Args:
        master: `String` representation of the TensorFlow master to use.
        init_op: Optional `Operation` used to initialize the model.
        saver: A `Saver` object used to restore a model.
        checkpoint_dir: Path to the checkpoint files. The latest checkpoint in
          the dir will be used to restore.
        checkpoint_filename_with_path: Full file name path to the checkpoint
          file.
        wait_for_checkpoint: Whether to wait for checkpoint to become available.
        max_wait_secs: Maximum time to wait for checkpoints to become available.
        config: Optional `ConfigProto` proto used to configure the session.
        init_feed_dict: Optional dictionary that maps `Tensor` objects to feed
          values.  This feed dictionary is passed to the session `run()` call
          when running the init op.
        init_fn: Optional callable used to initialize the model. Called after
          the optional `init_op` is called.  The callable must accept one
          argument, the session being initialized.
      Returns:
        A `Session` object that can be used to drive the model.
      Raises:
        RuntimeError: If the model cannot be initialized or recovered.
        ValueError: If both checkpoint_dir and checkpoint_filename_with_path are
          set.
    """
    sess = tf.Session(master, graph=self._graph, config=config)
    if init_op is None and not init_fn and self._local_init_op is None:
      raise RuntimeError("Model is not initialized and no init_op or "
                         "init_fn or local_init_op was given")
    if init_op is not None:
      sess.run(init_op, feed_dict=init_feed_dict)
    if init_fn:
      init_fn(sess)
    sess.run(tf.local_variables_initializer()) # why do i have to add this?
    print("LOCAL INIT OP", self._local_init_op)
    sess, is_loaded_from_checkpoint = self._restore_checkpoint(
        master,
        sess,
        saver,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename_with_path=checkpoint_filename_with_path,
        wait_for_checkpoint=wait_for_checkpoint,
        max_wait_secs=max_wait_secs,
        config=config,
        load_fc=load_fc,
        assign_ops=assign_ops,
        restore_dict=restore_dict)


    local_init_success, msg = self._try_run_local_init_op(sess)
    if not local_init_success:
      raise RuntimeError(
          "Init operations did not make model ready for local_init.  "
          "Init op: %s, init fn: %s, error: %s" % (_maybe_name(init_op),
                                                   init_fn,
                                                   msg))

    is_ready, msg = self._model_ready(sess)
    if not is_ready:
      raise RuntimeError(
          "Init operations did not make model ready.  "
          "Init op: %s, init fn: %s, local_init_op: %s, error: %s" %
          (_maybe_name(init_op), init_fn, self._local_init_op, msg))
    return sess

def _restore_embed(embed_var, var_to_shape_map, reader):
  if len([var for var in var_to_shape_map if 'EmbeddingMatrix' in var]) > 0:
    return None, None # assume same name
  for var in var_to_shape_map:
    if (var.endswith('dense/kernel')
        and var_to_shape_map[var] == tf.transpose(embed_var).shape):
      print('Assigning', var, 'to', embed_var.name)
      tensor = reader.get_tensor(var).T
      if tensor.dtype != var.dtype.as_numpy_dtype():
        return embed_var.assign(tf.cast(tensor, embed_var.dtype)), True
      return embed_var, False
  return None, None

def get_assign_ops_and_restore_dict(filename, restore_all=False):
  """Helper function to read variable checkpoints from filename.
  Iterates through all vars in restore_all=False else all trainable vars. It
  attempts to match variables by name and variable shape. Returns a possibly
  empty list of assign_ops, and a possibly empty dictionary for tf.train.Saver()
  """
  def check_name_and_shape(name, var, shape_map):
    if name in shape_map:
      # Cannot check variables with unknown sizes such as cudnn rnns
      if str(var.shape) == "<unknown>":
        # Just return True and hope the shapes match
        return True
      if var.shape == shape_map[name]:
        return True
    return False

  assign_ops = []
  restore_dict = {}

  try:
    reader = tf.train.NewCheckpointReader(filename)
    var_to_shape_map = reader.get_variable_to_shape_map()

    variables = tf.trainable_variables()
    if restore_all:
      variables = tf.get_collection(tf.GraphKeys.VARIABLES)
    for var in variables:
      idx = var.name.find(":")
      if idx != -1:
        true_name = var.name[:idx]
      loss_idx = re.search("Loss_Optimization", true_name)
      if 'EmbeddingMatrix' in true_name:
        embed_restore, assign = _restore_embed(var, var_to_shape_map, reader)
        if assign:
          assign_ops.append(embed_restore)
        else:
          restore_dict[true_name] = embed_restore
      if check_name_and_shape(true_name, var, var_to_shape_map):
        tensor = reader.get_tensor(true_name)
        if tensor.dtype != var.dtype.as_numpy_dtype():
          assign_ops.append(var.assign(tf.cast(tensor, var.dtype)))
        else:
          restore_dict[true_name] = var
      elif loss_idx:
        loss_idx = loss_idx.end()
        if FP32_TEST.search(true_name):
          true_name = FP32_TEST.sub("", true_name)
        else:
          true_name = (true_name[:loss_idx]
                       + "/Loss_Optimization/FP32-master-copy"
                       + true_name[loss_idx:])
        if check_name_and_shape(true_name, var, var_to_shape_map):
          tensor = reader.get_tensor(true_name)
          if tensor.dtype != var.dtype.as_numpy_dtype():
            assign_ops.append(var.assign(tf.cast(tensor, var.dtype)))
          else:
            restore_dict[true_name] = var
      else:
        print("Not restoring {}".format(var.name))
        if true_name not in var_to_shape_map:
          print("true name [{}] was not in shape map".format(true_name))
        else:
          if var.shape != var_to_shape_map[true_name]:
            print(("var.shape [{}] does not match var_to_shape_map[true_name]"
                   "[{}]").format(var.shape, var_to_shape_map[true_name]))
        print("WARNING: Run will mostly error out due to this")
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")
    if ("Data loss" in str(e) and
        (any([e in filename for e in [".index", ".meta", ".data"]]))):
      proposed_file = ".".join(filename.split(".")[0:-1])
      v2_file_error_template = """
      It's likely that this is a V2 checkpoint and you need to provide the
      filename *prefix*.  Try removing the '.' and extension.  Try:
      inspect checkpoint --file_name = {}"""
      print(v2_file_error_template.format(proposed_file))
    raise ValueError("Error in loading checkpoint")
  return assign_ops, restore_dict

def run_assign_and_saver(sess, filename, assign_ops, restore_dict):
  """Helper function to restore variables. All variables with the same dtype
  can be restored using tf.train.Saver(). All variables with different dtype
  are restored using assign_ops
  """
  if restore_dict:
    restorer = tf.train.Saver(restore_dict)
    restorer.restore(sess, filename)
  if assign_ops:
    sess.run(assign_ops)

def _maybe_name(obj):
  """Returns object name if it has one, or a message otherwise.
  This is useful for names that apper in error messages.
  Args:
    obj: Object to get the name of.
  Returns:
    name, "None", or a "no name" message.
  """
  if obj is None:
    return "None"
  elif hasattr(obj, "name"):
    return obj.name
  else:
    return "<no name for %s>" % type(obj)
