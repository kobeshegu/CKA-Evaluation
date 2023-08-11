python -u calc_metrics.py \
       --metrics=mixermlp_fid50k_full \
       --data ../ffhq256x256.zip \
       --eval_bs=1000 \
       --layers=blocks.0,blocks.1,blocks.2,blocks.3,blocks.4,blocks.5,blocks.6,blocks.7,blocks.8,blocks.9,blocks.10,blocks.11 \
       --mirror=1 \
       --cache=0 \
       --cfg=stylegan2 \
       --random=0 \
       --feature_save_flag=0 \
       --max_real=70000 \
       --num_gen=49999 \
       --save_name=ffhq_50K_vs_ffhq_49K_randomset \
       --save_stats ./statsmixer \
       --generate ../random_49K_ffhq.zip \
       --save_res ./results \

python -u grad_cam.py \
        --detectors=mixer_b16_224 \
        --stats_path ./statsmixer \
        --html_name=mixermlp \
        --generate_image_path ../ffhq_cam \
        --outdir ./grad_cam \


python -u calc_metrics.py \
       --metrics=gmlp_fid50k_full \
       --data ../ffhq256x256.zip \
       --eval_bs=1000 \
       --layers=blocks.0,blocks.1,blocks.2,blocks.3,blocks.4,blocks.5,blocks.6,blocks.7,blocks.8,blocks.9,blocks.10,blocks.11,blocks.12,blocks.13,blocks.14,blocks.15,blocks.16,blocks.17,blocks.18,blocks.19,blocks.20,blocks.21,blocks.22,blocks.23,blocks.24,blocks.25,blocks.26,blocks.27,blocks.28,blocks.29 \
       --mirror=1 \
       --cache=0 \
       --cfg=stylegan2 \
       --random=0 \
       --feature_save_flag=0 \
       --max_real=70000 \
       --num_gen=49999 \
       --save_name=ffhq_50K_vs_ffhq_49K_randomset \
       --save_stats ./statsgmlp \
       --generate ../random_49K_ffhq.zip \
       --save_res ./results \

python -u grad_cam.py \
        --detectors=mixer_b16_224 \
        --stats_path ./statsgmlp \
        --html_name=gmlp_s16_224 \
        --generate_image_path ../ffhq_cam \
        --outdir ./grad_cam \

sendemail -f mpyang_ecust@163.com -t mpyang_ecust@163.com -s smtp.163.com -u "fwy Code Finished!" -o message-content-type=html -o message-charset=utf-8 -xu mpyang_ecust -xp WECCVEFLPVBMVCFK -m "Your code has finished, My Honor to serve you, Sir."