checkpoint_path=./checkpoints/wmt-en2de-baseline
data_bin=/home/duantea/data/wmt-bin/

beam=4
num=1


ff="valid test"

for f in $ff; do
    CUDA_VISIBLE_DEVICES=$num python fairseq_cli/generate.py $data_bin \
        --path $checkpoint_path \
        --gen-subset $f \
        --lenpen 0.6 \
        --beam $beam \
        --remove-bpe | tee out-wmt/$f.gen
done