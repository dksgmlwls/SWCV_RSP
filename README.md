# [SW중심대학 공동 AI 경진대회 2023 - SWCV]RSP

‼️환경설정‼️
1. apt update
2. pt install git
3. git clone https://github.com/ViTAE-Transformer/RSP.git
4. pip install -U openmim
5. apt install gcc g++
6. mim install mmengine
7. mim install mmcv==1.4.4
8. apt install libxml2
9. wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
10. sh cuda_11.1.0_455.23.05_linux.run
11. accept 입력
12. install 선택
13. pip install mmcv-full==1.4.4 -f https://download.openmlab.com/mmcv/dist/cu111/torch1.8/index.html
14. cd RSP/SemanticSegmentation
15. pip install -v -e .
16. apt-get install ffmpeg libsm6 libxext6 -y
17. pip install timm

‼️train 실행코드‼️
1. cd RSP/SemanticSegmentation
2. python -m torch.distributed.launch --nproc_per_node=3 --master_port=40001 tools/train.py configs/vitae_win/custom_dataset_renew_28k.py --work-dir /mount/SSD_2T_a/SWCV_submit/RSP/SemanticSegmentation/work_dirs/custom_dataset_renew_28k/ --launcher 'pytorch'

‼️inference 실행코드‼️
1. cd RSP/SemanticSegmentation
2. python tools/test.py configs/vitae_win/custom_dataset_renew_28k.py /mount/SSD_2T_a/SWCV_submit/RSP/SemanticSegmentation/work_dirs/custom_dataset_renew_28k/iter_32000.pth --show-dir /mount/SSD_2T_a/SWCV_submit/mask_dir/RSP/
