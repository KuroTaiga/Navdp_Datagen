# testing NAS
SCENE=test_scene
mkdir -p /mnt/nas/path_video_frames_Jiankun_test/$SCENE \
  && : > /mnt/nas/path_video_frames_Jiankun_test/$SCENE/__touch_test__ \
  && rm -f /mnt/nas/path_video_frames_Jiankun_test/$SCENE/__touch_test__ \
  && echo "OK: can write to NAS" || echo "FAIL: cannot write to NAS"
# run datagen example cal
conda activate cuda121
python render_label_paths.py \
  --overwrite --stabilize \
  --actor-seq-dir /media/dongjk/walk_45/ \
  --height-offset -0.098 \
  --minimal-frames 90 \
  --show-BEV \
  --gpu-only \
  --offload-nas-dir /mnt/nas/jiankundong/path_video_frames_Jiankun_test \
  --offload-min-free-gb 0.5
