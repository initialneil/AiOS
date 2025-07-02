#!/bin/bash
CHECKPOINT=$1
INPUT_VIDEO=$2
OUTPUT_DIR=$3
GPU_NUM=${4:-8}
python -m torch.distributed.launch \
    --nproc_per_node ${GPU_NUM} \
    main.py \
    -c "config/aios_smplx_youtube.py" \
    --options batch_size=8 backbone="resnet50" \
    --resume ${CHECKPOINT} \
    --eval \
    --inference \
    --inference_input ${INPUT_VIDEO} \
    --output_dir demo/${OUTPUT_DIR}

# Example usage:
# python -m torch.distributed.launch --nproc_per_node 1 main.py",
# "-c", "config/aios_smplx_youtube.py", 
# "--options", 
# "batch_size=6",
# "--resume", "data/checkpoint/aios_checkpoint.pth", 
# "--eval", 
# "--inference", 
# // "--to_vid", 
# "--inference_input", 
# "/mnt/f/Datasets/YoutubeGestureDataset/frame_segments_2fps", 
# "--output_dir", 
# "/mnt/f/Datasets/YoutubeGestureDataset/AiOS",
