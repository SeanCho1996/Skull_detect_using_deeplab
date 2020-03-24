python vis.py ^
    --logtostderr ^
    --vis_split="val" ^
    --model_variant="xception_65" ^
    --atrous_rates=6 ^
    --atrous_rates=12 ^
    --atrous_rates=18 ^
    --output_stride=16 ^
    --decoder_output_stride=4 ^
    --vis_crop_size=801,801 ^
    --dataset="skull_detect" ^
    --checkpoint_dir=.\model ^
    --vis_logdir=./vis_res ^
    --dataset_dir=./datasets/tfrecord
	--max_number_of_iterations=1


PAUSE