checkpoint_path=./checkpoints/ldc-ch2en-baseline/checkpoint_best.pt
bleu=/data1/cjin/data/multi-bleu.perl

data_bin=/data1/cjin/data/ldc/data-bin/
dir=/data1/cjin/data/ldc/


beam=5
num=1
batch=100
tokens=10000

src=ch
tgt=en


ff="nist02 nist03 nist04 nist05 nist08"

for f in $ff; do
    CUDA_VISIBLE_DEVICES=$num python fairseq_cli/interactive.py $data_bin \
        --path $checkpoint_path \
        --beam $beam \
        --buffer-size $batch \
        --max-tokens $tokens \
        --lenpen 1 \
        --remove-bpe \
        -s $src -t $tgt < $dir/$f.bpe.in > dumped/$f.$tgt.tmp
    grep ^H dumped/$f.$tgt.tmp | cut -f3- > dumped/$f.$tgt
    perl $bleu -lc $dir/$f.ref.* < dumped/$f.$tgt > $f.score
done 



echo "ch2en-baseline" >> ch2en.score
for f in $ff;do
    cat $f.score >> ch2en.score
    rm $f.score
done
