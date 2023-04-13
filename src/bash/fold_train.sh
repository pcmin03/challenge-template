for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python ../train.py data.val_fold="$i"
done