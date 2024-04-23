export CUDA_VISIBLE_DEVICES=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CUBLAS_WORKSPACE_CONFIG=:4096:8
CUBLAS_WORKSPACE_CONFIG=:16:8
PYTHONHASHSEED=0

tensorboard --logdir=logs

python train.py --config configs/h36m/MotionAGFormer-base-sh.yaml --eval-only --checkpoint /data-home/checkpoints/motionagformer-b-h36m-sh.pth.tr
python train.py --config configs/h36m/MotionAGFormer-base-cpn.yaml --eval-only --checkpoint /data-home/checkpoints/motionagformer-b-h36m-cpn.pth.tr
python train.py --config configs/h36m/MotionAGFormer-base-hrnet.yaml --eval-only --checkpoint /data-home/checkpoints/motionagformer-b-h36m-hrnet.pth.tr
python train.py --config configs/h36m/MotionAGFormer-base-gt.yaml --eval-only --checkpoint /data-home/checkpoints/motionagformer-b-h36m-gt.pth.tr

python train.py --config configs/h36m/MotionBERT-sh.yaml --eval-only --checkpoint /data-home/checkpoints/motionbert-finetune-h36m-author.pth.tr
python train.py --config configs/h36m/MotionBERT-sh.yaml --eval-only --checkpoint /data-home/checkpoints/motionbert-scratch-h36m-sh-author.pth.tr
python train.py --config configs/h36m/MotionBERT-gt.yaml --eval-only --checkpoint /data-home/checkpoints/motionbert-scratch-h36m-gt.pth.tr

python train_app.py --config configs/h36m/AdaptivePosePoolingv2-hrnet.yaml

torchrun --nproc_per_node=2 --master_port=11456 train_app.py --config configs/h36m/AdaptivePosePoolingv3-sh.yaml
torchrun --nproc_per_node=2 --master_port=11454 train_app.py --config configs/h36m/AdaptivePosePoolingv2-sh.yaml
torchrun --nproc_per_node=2 --master_port=11455 train_app.py --config configs/h36m/AdaptivePosePoolingv2-gt.yaml

conda install python==3.8.18 -y
mamba install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
mamba install tqdm PyYAML scikit-learn scikit-image yacs numba prettytable diffusers transformers -y
pip install torchprofile matplotlib easydict einops filterpy timm wandb opencv-python fvcore
apt update && apt install tmux -y

python train_app_3dhp.py --config configs/mpi/AdaptivePosePoolingv2.yaml
python train_app_3dhp.py --config configs/mpi/AdaptivePosePoolingv2-motionbert-scratch.yaml

python train_app_3dhp.py --config configs/mpi/AdaptivePosePoolingv2-motionbert-scratch.yaml --checkpoint /data-home/checkpoints/resume_motionbert.tr --resume
python train_app_3dhp.py --config configs/mpi/AdaptivePosePoolingv2.yaml --checkpoint /data-home/checkpoints/resume_motionagformer.tr --resume

python train.py --config configs/h36m/Transformer/Transformer-st-st-sh.yaml
python train.py --config configs/h36m/Transformer/Transformer-ts-ts-sh.yaml
python train.py --config configs/h36m/Transformer/Transformer-para-para-sh.yaml

python demo_app/vis.py --video sample_video.mp4 --gpu 3
python demo_app/vis_app.py --video sample_video.mp4 --gpu 4
python demo_app/vis_motionbert.py --video sample_video.mp4 --gpu 3
python demo_app/vis_app_motionbert.py --video sample_video.mp4 --gpu 4

python demo_app/vis.py --video sample_video_2.mp4 --gpu 3
python demo_app/vis_app.py --video sample_video_2.mp4 --gpu 4
python demo_app/vis_motionbert.py --video sample_video_2.mp4 --gpu 3
python demo_app/vis_app_motionbert.py --video sample_video_2.mp4 --gpu 4

python demo_app/vis.py --video sample_video_3.mp4 --gpu 3
python demo_app/vis_app.py --video sample_video_3.mp4 --gpu 4
python demo_app/vis_motionbert.py --video sample_video_3.mp4 --gpu 3
python demo_app/vis_app_motionbert.py --video sample_video_3.mp4 --gpu 4

python train_app.py --config configs/h36m/AdaptivePosePoolingv2-hot-sh.yaml
python train_app.py --config configs/h36m/AdaptivePosePoolingv2-mixste-cpn.yaml