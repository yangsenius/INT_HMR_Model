#!/bin/bash
# smplx: view operation requires contiguous tensor, replacing by reshape operation
sed -i "347c   rel_joints.reshape(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)" ~/anconda3/envs/int_hmr/lib/python3.6/site-packages/smplx/lbs.py

sed -i "96,97d" ~/anconda3/envs/int_hmr/lib/python3.6/site-packages/horovod/torch/mpi_ops.py

unset OMPI_MCA_plm_rsh_agent
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export LANG=zh_CN.UTF-8

date=`date +%Y%m%d_%H%M%S`
export LANG=en_US.UTF-8

# link to the work dir to save checkpoints and logs
if [ ! -d 'workdir' ];then
    mkdir -p workdir
fi
work_dir=workdir/token3d_training_dir
model_name=hvd_token3d_phase1
exp_dir=${work_dir}/${model_name}
resume_name='checkpoint.pt'

resume_from=$exp_dir/$resume_name
best_from=$work_dir/$best_name

if [ ! -d ${exp_dir} ];then
  mkdir -p ${exp_dir}
fi
echo 'current work dir is: '${exp_dir}

# for example
# To run on a machine with 4 GPUs:
# horovodrun -np 4 -H localhost:4 python train.py

# To run on 4 machines with 4 GPUs each
# horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py


gpu_min=$1 # total_min_gpu_num
node_list=$2 #server1_ip:gpu_num,server2_ip:gpu_num

horovodrun -np ${gpu_min} -H ${node_list} \
    python train_hvd.py --cfg configs/baseline_phase1.yaml\
    --resume $resume_from \
    --logdir ${exp_dir} 2>&1 | tee -a  ${exp_dir}/hvd_output.log
