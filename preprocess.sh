python resize.py \
  --input datasets/test_vest/test/image \
  --output datasets/test_vest/test/image \
  --width 576 \
  --height 768

python extract_densepose.py \
  --config detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
  --model detectron2/projects/DensePose/ckpt/model_final_162be9.pkl \
  --input datasets/test_vest/test/image \
  --output_dir datasets/test_vest/test/image-densepose \
  --width 576 \
  --height 768


python extract_upper_cloth.py \
  --input_dir datasets/test_vest/test/image \
  --output_dir datasets/test_vest/test/agnostic-mask \
  --densepose_dir datasets/test_vest/test/image-densepose \
  --naturality 1 \
  --ratio 0.1 \
  --vest 1 \
  --vest_padding 10
