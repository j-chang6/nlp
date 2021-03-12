checkpoint_path=./checkpoints/iwslt14-de2en-baseline
data_bin=/data1/cjin/data/iwst14/de2en/data-bin

beam=5
num=1


ff="valid test"

for f in $ff; do
    CUDA_VISIBLE_DEVICES=$num python fairseq_cli/generate.py $data_bin \
        --path $checkpoint_path \
        --gen-subset $f \
        --beam $beam \
        --remove-bpe | tee out-iwslt/$f.gen
done