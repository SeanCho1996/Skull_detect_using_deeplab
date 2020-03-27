python export_model.py ^
    --logtostderr ^
    --checkpoint_path=./model/model.ckpt-1000 ^
    --export_path=./XOut\frozen_inference_graph.pb ^
    --model_variant="xception_65" ^
    --atrous_rates=6 ^
    --atrous_rates=12 ^
    --atrous_rates=18 ^
    --output_stride=16 ^
    --decoder_output_stride=4 ^
    --num_classes=2 ^
    --inference_scales=1.0

PAUSE