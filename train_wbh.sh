CUDA_VISIBLE_DEVICES=0,1,2,3 python train_distributed.py \
 --dataset mcse \
 --batch_size 4 \
 --num_workers 16 \
 --mics 8 \
 --M 8 \
 --checkpoint_dir "/data/wbh/l3das23/experiment/4gpu"\
 --results_path "/data/wbh/l3das23/experiment/4gpu"\
 --saving_interval 0.25\
 --valid_interval 0.25