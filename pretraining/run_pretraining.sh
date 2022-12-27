export CUDA_VISIBLE_DEVICES=0
python pretraining/run_pretraining.py \
    --init_from_checkpoint \
    --pretrained_model_name_or_path=pretrained_model/splinter \
    --train_data_file=data/pretrain_data/pretrain_data_1.pkl,data/pretrain_data/pretrain_data_2.pkl \
    --batch_size=48 \
    --scheduler=linear \
    --learning_rate=1e-5 \
    --save_steps=2000 \
    --logging_steps=200 \
    --weight_decay=0.01 \
    --epochs=1 \
    --overwrite_output_dir \
