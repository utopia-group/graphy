#!/bin/bash

mode=$1
addtional_args=$2

dataset="datavis"
held_out="True"
test_set="car"
bert_model="bert-large-uncased"
plot_epochs="30"
plot_lr="1.5e-05"
plot_wr="0.1"
color_epochs="20"
color_lr="2e-05"
color_wr="0.06"
column_epochs="20"
column_lr="2e-05"
column_wr="0.06"
count_epochs="20"
count_lr="1e-05"
count_wr="0.05"
mean_epochs="30"
mean_lr="2e-05"
mean_wr="0.05"
sum_epochs="20"
sum_lr="2e-05"
sum_wr="0.05"

input_dim="100"
hidden_size="150"
max_field_num="10"
max_token_num_per_field="5"
batch_size="16"

held_out_flag=""
held_out_str=""
if [[ $held_out = *"True"* ]]; then
    held_out_flag="--held_out"
    held_out_str=".held_out"
fi

ts=`date "+%Y%m%d-%H%M%S"`

ins_id="$dataset.$test_set.$bert_model.$batch_size.$input_dim.$hidden_size.$max_field_num.$max_token_num_per_field.p$plot_epochs.$plot_lr.$plot_wr.cr$color_epochs.$color_lr.$color_wr.cn$column_epochs.$column_lr.$column_wr.ct$count_epochs.$count_lr.$count_wr.m$mean_epochs.$mean_lr.$mean_wr.s$sum_epochs.$sum_lr.$sum_wr$held_out_str"
log_file_name="$mode.$ins_id.$ts.log"

model_dir="model/$ins_id"

CUDA_VISIBLE_DEVICES=0
cmd="python3 experiment.py --$mode \
                                                --dataset $dataset \
                                                --test_set $test_set \
                                                --bert_model $bert_model \
                                                $held_out_flag \
                                                --plot_epochs $plot_epochs \
                                                --plot_lr $plot_lr \
                                                --plot_wr $plot_wr \
                                                --color_epochs $color_epochs \
                                                --color_lr $color_lr \
                                                --color_wr $color_wr \
                                                --column_epochs $column_epochs \
                                                --column_lr $column_lr \
                                                --column_wr $column_wr \
                                                --count_epochs $count_epochs \
                                                --count_lr $count_lr \
                                                --count_wr $count_wr \
                                                --mean_epochs $mean_epochs \
                                                --mean_lr $mean_lr \
                                                --mean_wr $mean_wr \
                                                --sum_epochs $sum_epochs \
                                                --sum_lr $sum_lr \
                                                --sum_wr $sum_wr \
                                                --input_dim $input_dim \
                                                --hidden_size $hidden_size \
                                                --max_field_num $max_field_num\
                                                --max_token_num_per_field $max_token_num_per_field \
                                                --batch_size $batch_size \
                                                --ts $ts > $log_file_name"

echo "run $cmd"
eval $cmd

mv $log_file_name "$model_dir/${mode}_${ts}.log"

if [ $# -ge 2 ]
then
    mv $model_dir "${model_dir}_${additional_args}"
else
    :
fi
