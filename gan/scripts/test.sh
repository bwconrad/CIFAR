#!/bin/bash
python test.py \
--batch_size 256 \
--dataset C10 \
--ema --use_ema \
--weights_root ./weights \
--logs_root ./logs \
--samples_root ./samples \
--mh_loss \
--model=BigGANmh \
--G_eval_mode \
--experiment_name CIFAR10 \
--load_weights '064000' \
--sample_sheets \
#--sample_random \
#--sample_interps \

