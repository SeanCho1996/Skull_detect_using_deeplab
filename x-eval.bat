python eval.py ^
  --logtostderr ^
  --eval_split="val" ^
  --model_variant="xception_65" ^
  --atrous_rates=6 ^
  --atrous_rates=12 ^
  --atrous_rates=18 ^
  --output_stride=16 ^
  --decoder_output_stride=4 ^
  --dataset="skull_detect" ^
  --checkpoint_dir="./model" ^
  --eval_logdir="./eval_res" ^
  --dataset_dir="./datasets/tfrecord" ^
  --max_number_of_evaluations=1
PAUSE