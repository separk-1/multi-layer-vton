python extract_densepose.py \
  --config detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
  --model detectron2/projects/DensePose/ckpt/model_final_162be9.pkl \
  --input datasets/test_joon/test/image/joon.jpg \
  --output_dir datasets/test_joon/test/image-densepose \
  --width 576 \
  --height 768


python extract_upper_cloth.py \
  --input datasets/test_joon/test/image/joon.jpg \
  --output_jpg datasets/test_joon/test/agnostic-mask/joon.jpg \
  --output_png datasets/test_joon/test/agnostic-mask/joon_mask.png \
  --densepose datasets/test_joon/test/image-densepose/I.npy \
  --naturality 1 \
  --ratio 0.1 \
  --vest 1 \
  --vest_padding 10
