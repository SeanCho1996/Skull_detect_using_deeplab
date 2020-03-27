python train.py  ^
    --logtostderr  ^
    --training_number_of_steps=500  ^
    --train_split="train"  ^
    --model_variant="xception_65"  ^
    --atrous_rates=6  ^
    --atrous_rates=12  ^
    --atrous_rates=18  ^
    --output_stride=16  ^
    --decoder_output_stride=4  ^
    --train_crop_size=801,801  ^
    --train_batch_size=1 ^
    --fine_tune_batch_norm=False ^
    --dataset="skull_detect" ^
    --tf_initial_checkpoint=.\deeplabv3_pascal_trainval\model.ckpt ^
    --train_logdir=./model ^
    --dataset_dir=./datasets/tfrecord ^
    --initialize_last_layer=False ^
    --last_layers_contain_logits_only=True
PAUSE