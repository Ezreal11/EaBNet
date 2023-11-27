if [ "$1" == "" ]; then
    echo positional argument experiment_name is required
    exit 1
fi
exp_root=/data2/zhouchang/experiments/eabnet/$1
if [ -d "$exp_root" ]; then
    read -p "$exp_root already exists. delete it? (y/n):" response

    if [[ $response == "y" ]]; then
        rm -rf $exp_root
        echo delete dir $exp_root
    else
        echo bye
        exit 1
    fi
fi

mkdir -p $exp_root
echo create dir $exp_root

CUDA_VISIBLE_DEVICES=2 python train_distributed.py \
 --dataset mcse \
 --batch_size 8 \
 --num_workers 24 \
 --mics 8 \
 --M 8 \
 --results_path "$exp_root/results" \
 --checkpoint_dir "$exp_root/checkpoints"