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
 --batch_size 8 \
 --num_workers 32 \
 --valid_interval 1 \
 --saving_interval 0.25 \
 --mics 9 \
 --M 9 \
 --results_path "$exp_root/results" \
 --checkpoint_dir "$exp_root/checkpoints" \
 --exp_root $exp_root \
 --mcse_dataset_train_set online \
 --mcse_dataset_val_set data/datasets/mcse_val_setting2 \
 --mcse_dataset_settings dataset/mcse_dataset_settings_v2.json \
#  --validate_once_before_train