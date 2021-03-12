data_dir=/data1/cjin/data/wmt/wmt17_en_de/
dir=/data1/cjin/data/wmt/wmt17_en_de/data-bin

preprocess(){

python fairseq_cli/preprocess.py \
        --trainpref $data_dir/train \
        --validpref $data_dir/valid \
        --testpref $data_dir/test \
        --source-lang en --target-lang de \
        --destdir  $dir \
        --nwordssrc 32768 --nwordstgt 32768 \
        --joined-dictionary \
        --workers 30 \

}

preprocess