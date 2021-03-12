data_dir=/data1/cjin/data/iwst14/de2en
dir=/data1/cjin/data/iwst14/de2en/data-bin

preprocess(){

python fairseq_cli/preprocess.py \
        --trainpref $data_dir/train \
        --validpref $data_dir/valid \
        --testpref $data_dir/test \
        --source-lang de --target-lang en \
        --destdir  $dir \
        --workers 30 \

}

preprocess