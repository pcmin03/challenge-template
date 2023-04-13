for i in {0..4}
do
    HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python ../train.py \
                            'data=frog_sign.yaml' \
                            'model=frog_sign.yaml' \
                            data.dataset_cfg.val_fold="$i"
done