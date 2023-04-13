for i in {0..4}
do
    HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python ../train.py \
                            'model=arcface.yaml' \
                            'data.csv_path=/opt/sign/data/sign_data/asl-signs/fold_train.csv' \
                            'data.npy_name=X.npy' \
                            'data.lab_name=y.npy'
                            data.dataset_cfg.val_fold="$i"
done