for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python ../train.py \
                            data.val_fold="$i" \
                            'model=arcface.yaml' \
                            'data.csv_path=/opt/sign/data/sign_data/asl-signs/fold_train.csv' \
                            'data.npy_name=small_feature_data.npy' \
                            'data.lab_name=small_feature_labels.npy'
                            
done