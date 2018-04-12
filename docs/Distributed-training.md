We implemented an experimental support for distributed training using [Uber's Horovod project](https://github.com/uber/horovod).

You can use this training mode if you have: a) one machine with mutiple-gpus and/or b) many machines with (potentially more than one) GPUs.


## Requirements
* Tensorflow 1.4 (1.2 and 1.3 migh work too)
* Python 2.7 or 3.6
* [Uber's Horovod](https://github.com/uber/horovod)
* [NCCL](https://developer.nvidia.com/nccl) if you want to make use of NVLINK

## Training on one machine with several GPUs

### Prepare environment
#### 1. Install NCCL
  * (Might be already installed in NVIDIA's docker images for Tensorflow)
  * You can skip this step if you don't have NVLINK
#### 2. Make sure Tensorflow is installed
#### 3. Install OpenMPI:
  * (Might be already installed in NVIDIA's docker images for Tensorflow)
  * ```wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz```.
  * ```tar -xvf openmpi-3.0.0.tar.gz```
  * ```cd openmpi-3.0.0```
  * ```./configure --prefix=/usr/local```
  * ```make all install``` Or you might need sudo: ```sudo make all install```
  * ```ldconfig```
#### 4. Install [Horovod](https://github.com/uber/horovod/blob/master/docs/gpus.md)
  * (Might be already installed in NVIDIA's docker images for Tensorflow)
  * Run ```locate libnccl``` to see NCCL is installed. In my case it is ```/usr/lib/x86_64-linux-gnu```
  * Run ``` HOROVOD_NCCL_HOME=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod```


### Running on single machine with many GPUs

1. Edit your ".json" config to:
  * Set `num_gpus` to `1`
  * Remove `source_file_eval` and `target_file_eval`. Currently running evaluation in parallel with training using Horovod is not supported.
2. Execute training command (in this example, machine has 8 GPUs):
```
mpirun --mca orte_base_help_aggregate 0 -mca btl ^openib \
-np 8 -H localhost:8 \
-bind-to none -map-by slot -x LD_LIBRARY_PATH python /home/okuchaiev/repos/OpenSeq2Seq/run_dist.py \
--config_file=/mnt/shared/okuchaiev/Workspace/NMT/OpenSeq2Seq/GNMT-like/HolidayRuns/hvd8x_1xGNMT-like-adam-lars1.0-init0.01-lr_8*0.0008_decay_policy_1.json \ 
--logdir=/mnt/shared/okuchaiev/Workspace/NMT/OpenSeq2Seq/GNMT-like/HolidayRuns/hvd8x_1xGNMT-like-adam-lars1.0-init0.01-lr_8*0.0008_decay_policy_1 \
--summary_frequency=50 --max_steps=57142
```
or, if you are running from inside Docker image as root:
```
mpirun --allow-run-as-root --mca orte_base_help_aggregate 0 -mca btl ^openib \
-np 8 -H localhost:8 \
-bind-to none -map-by slot -x LD_LIBRARY_PATH python /home/okuchaiev/repos/OpenSeq2Seq/run_dist.py \
--config_file=/mnt/shared/okuchaiev/Workspace/NMT/OpenSeq2Seq/GNMT-like/HolidayRuns/hvd8x_1xGNMT-like-adam-lars1.0-init0.01-lr_8*0.0008_decay_policy_1.json \ 
--logdir=/mnt/shared/okuchaiev/Workspace/NMT/OpenSeq2Seq/GNMT-like/HolidayRuns/hvd8x_1xGNMT-like-adam-lars1.0-init0.01-lr_8*0.0008_decay_policy_1 \
--summary_frequency=50 --max_steps=57142
```

### Running on many machines with many GPUs

1. Make sure, Tensorflow, OpenMPI and Horovod are installed and same versions on all machines
2. Setup [passwordless ssh](https://www.linuxbabe.com/linux-server/setup-passwordless-ssh-login) between all pairs of machines.
  * eval `ssh-agent -s'
  * ssh-add

## FAQ

#### 1. ```...The value of the MCA parameter "plm_rsh_agent" was set to a path that could not be found...```
  * Install SSH: ```apt install openssh-client openssh-server```, might need sudo: ```sudo apt install openssh-client openssh-server```