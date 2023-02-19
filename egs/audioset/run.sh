#!/bin/bash
set -x
export TORCH_HOME=../../pretrained_models

model=ast
transformer=$1
dataset=audioset
# full or balanced for audioset
set=balanced
imagenetpretrain=True
if [ $set == balanced ]
then
  bal=none
  lr=5e-5
  epoch=25
  tr_data=./data/datafiles/train_data.json
  lrscheduler_start=10
  lrscheduler_step=5
  lrscheduler_decay=0.5
  wa_start=6
  wa_end=25
else
  bal=bal
  lr=1e-5
  epoch=5
  tr_data=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/whole_train_data.json
  lrscheduler_start=2
  lrscheduler_step=1
  lrscheduler_decay=0.5
  wa_start=1
  wa_end=5
fi
te_data=./data/datafiles/eval_data.json
freqm=48
timem=192
mixup=0.5
# corresponding to overlap of 6 for 16*16 patches
fstride=10
tstride=10
batch_size=12

dataset_mean=-4.2677393
dataset_std=4.5689974
audio_length=1024
noise=False

metrics=mAP
loss=BCE
warmup=True
wa=True

python ./prep_audioset.py

exp_dir=./exp/test-${set}-${transformer}-f$fstride-t$tstride-p$imagenetpretrain-b$batch_size-lr${lr}-decoupe
if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --transformer ${transformer} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/class_labels_indices.csv --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end}
