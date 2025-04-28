python resize.py \
  --input datasets/test_cmu/test/image/woman.jpg \
  --output datasets/test_cmu/test/image/woman.jpg \
  --width 576 \
  --height 768

python extract_densepose.py \
  --config detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
  --model detectron2/projects/DensePose/ckpt/model_final_162be9.pkl \
  --input datasets/test_cmu/test/image/woman.jpg \
  --output_dir datasets/test_cmu/test/image-densepose \
  --width 576 \
  --height 768


python extract_upper_cloth.py \
  --input datasets/test_cmu/test/image/woman.jpg \
  --output_jpg datasets/test_cmu/test/agnostic-mask/woman.jpg \
  --output_png datasets/test_cmu/test/agnostic-mask/woman_mask.png \
  --densepose datasets/test_cmu/test/image-densepose/I.npy \
  --naturality 1 \
  --ratio 0.1 \
  --vest 1 \
  --vest_padding 10
