#!/bin/bash
set -e

savename="somename"
environment="GridWorld-v0"
num_subs=2
augment="True"
num_rollouts=1024
train_time=250
continue_iter="False"
warmup=40
load=""
save="test"
tag="feature_test"


pretrain=0
for number in {1..5}
do
	filename="${environment}_${tag}_augment_${augment}_batchsize_${num_rollouts}_${number}"
	python3 main.py --warmup_time $warmup --filename $filename --pretrain $pretrain --task $environment --num_subs $num_subs --augment $augment --num_rollouts $num_rollouts --train_time $train_time
wait
done

exit 0

