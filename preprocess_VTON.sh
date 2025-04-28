# Step 1: Resize all images in a folder
python resize.py \
  --input_dir datasets/VTON-HD/test/image \
  --output_dir datasets/VTON-HD/test/image \
  --width 576 \
  --height 768

# Step 2: Generate DensePose for all resized images
python extract_densepose.py \
  --config detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
  --model detectron2/projects/DensePose/ckpt/model_final_162be9.pkl \
  --input_dir datasets/VTON-HD/test/image \
  --output_dir datasets/VTON-HD/test/image-densepose \
  --width 576 \
  --height 768

# Step 3: Generate upper cloth masks for all resized images
python extract_upper_cloth.py \
  --input_dir datasets/VTON-HD/test/image \
  --output_dir datasets/VTON-HD/test/agnostic-mask \
  --densepose_dir datasets/VTON-HD/test/image-densepose \
  --naturality 1 \
  --ratio 0.1 \
  --vest 1 \
  --vest_padding 10
