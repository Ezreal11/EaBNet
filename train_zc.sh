if [ "$1" == "" ]; then
    echo positional argument experiment_name is required
    exit 1
fi
exp_root=data/experiments/eabnet/$1
if [ -d "$exp_root" ]; then

    read -p "continue training? (y/n):" response

    if [[ $response == "y" ]]; then
        echo ok
    else
        read -p "$exp_root already exists. delete it? (y/n):" response
        if [[ $response == "y" ]]; then
            rm -rf $exp_root
            echo delete dir $exp_root
            mkdir -p $exp_root
            echo create dir $exp_root
        else
            echo bye
            exit 1
        fi
    fi
fi


CUDA_VISIBLE_DEVICES=1 python train_distributed.py \
 --dataset mcse \
 --batch_size 16 \
 --num_workers  24 \
 --valid_interval 1 \
 --mics 8 \
 --M 8 \
 --results_path "$exp_root/results" \
 --checkpoint_dir "$exp_root/checkpoints" \
 --mcse_dataset_train_set offline \
#  --validate_once_before_train