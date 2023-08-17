echo "begins..."
timer_start=`date "+%Y-%m-%d %H:%M:%S"`
echo "start time: $timer_start"

python -u calc_metrics.py \
       --metrics=vitcls_base_fid50k_full \
       --data ../ffhq256x256.zip \
       --eval_bs=1000 \
       --layers=blocks.0,blocks.1,blocks.2,blocks.3,blocks.4,blocks.5,blocks.6,blocks.7,blocks.8,blocks.9,blocks.10,blocks.11 \
       --mirror=1 \
       --cache=0 \
       --cfg=stylegan2 \
       --random=0 \
       --feature_save_flag=0 \
       --max_real=70000 \
       --num_gen=50000 \
       --save_name=testtime \
       --generate ../random_50K_ffhq.zip.zip \
       --save_res ./results \

timer_end=`date "+%Y-%m-%d %H:%M:%S"`
echo "end time：$timer_end"

start_seconds=$(date --date="$timer_start" +%s);
end_seconds=$(date --date="$timer_end" +%s);
echo "total time: "$((end_seconds-start_seconds))“s”