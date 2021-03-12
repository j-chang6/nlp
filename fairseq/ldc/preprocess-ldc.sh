data_dir=/data1/cjin/data/ldc
dir=/data1/cjin/data/ldc/data-bin

preprocess(){

python fairseq_cli/preprocess.py \
        --trainpref $data_dir/train.bpe \
        --validpref $data_dir/valid.bpe \
        --testpref $data_dir/test.bpe \
        --source-lang ch --target-lang en \
        --destdir  $dir \
        --workers 30 \

}

preprocess