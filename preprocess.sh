python resize.py \
  --input datasets/test_cmu/test/image \
  --output datasets/test_cmu/test/image \
  --width 576 \
  --height 768

python extract_densepose.py \
  --config detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
  --model detectron2/projects/DensePose/ckpt/model_final_162be9.pkl \
  --input datasets/test_cmu/test/image \
  --output_dir datasets/test_cmu/test/image-densepose \
  --width 576 \
  --height 768


python extract_upper_cloth.py \
  --input datasets/VTON-HD/test/image \
  --output_jpg datasets/VTON-HD/test/agnostic-mask \
  --output_png datasets/VTON-HD/test/agnostic-mask \
  --densepose datasets/VTON-HD/test/image-densepose \
  --naturality 1 \
  --ratio 0.1 \
  --vest 1 \
  --vest_padding 10
