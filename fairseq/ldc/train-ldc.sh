num=0,1,2,3
dropout=0.3
arch=transformer
max_tokens=4096
criterion=label_smoothed_cross_entropy
label_smoothing=0.1
lrscheduler=inverse_sqrt

save_dir=./checkpoints/ldc-ch2en-baseline
data_bin=/data1/cjin/data/ldc/data-bin


train(){
CUDA_VISIBLE_DEVICES=$num python fairseq_cli/train.py $data_bin \
            --optimizer adam \
            --stop-min-lr  1e-09 \
            --lr  0.0005\
            --clip-norm 0.0 \
            --criterion $criterion \
            --label-smoothing $label_smoothing \
            --lr-scheduler $lrscheduler \
            -s ch -t en \
            --dropout $dropout \
            --arch $arch \
            --warmup-init-lr 1e-07 \
            --warmup-updates 4000 \
            --weight-decay 0.0 \
            --adam-betas '(0.9, 0.98)' \
            --max-tokens $max_tokens \
            --save-dir $save_dir \
            --log-interval 100 \
            --no-progress-bar \
            --patience 10 \
            --eval-bleu \
            --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
            --eval-bleu-remove-bpe \
            --eval-bleu-detok moses \
            --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
            --keep-last-epochs 10 \
}

train
