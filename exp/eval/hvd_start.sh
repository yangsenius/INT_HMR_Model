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
model_name=hvd_token3d_phase3
config_yaml=configs/baseline_phase3.yaml


exp_dir=${work_dir}/${model_name}

if [ ! -d ${exp_dir} ];then
  mkdir -p ${exp_dir}
fi
echo 'current work dir is: '${exp_dir}

echo ">>>> eval"
#'epoch_100.pth.tar'   # 'model_best.pth.tar'   #44_9_model_best.pth.tar'
best_name='model_best.pth.tar'
best_from=$exp_dir/$best_name
#best_from='/cfs/cfs-31b43a0b8/personal/brucesyang/baseline_training_dir/tp_baseline_token3dpretrain/coco/transpose_r/token3dpretrain/checkpoint.pth'

python eval.py --cfg $config_yaml\
  --pretrained $best_from \
  --eval_ds 3dpw \
  --eval_set val \
  2>&1 | tee -a  ${exp_dir}/eval_output.log

python eval.py --cfg $config_yaml\
  --pretrained $best_from \
  --eval_ds 3dpw \
  --eval_set test \
  2>&1 | tee -a  ${exp_dir}/eval_output.log

python eval.py --cfg $config_yaml\
  --pretrained $best_from \
  --eval_ds h36m \
  --eval_set val \
  2>&1 | tee -a  ${exp_dir}/eval_output.log
