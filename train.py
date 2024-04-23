import argparse
import os
import sys

import numpy as np
import pkg_resources
import torch
import wandb
import shutil
import prettytable
from torch import optim
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext

from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity, loss_limb_var, loss_limb_gt, loss_angle, \
    loss_angle_velocity
from loss.pose3d import jpe as calculate_jpe
from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from loss.pose3d import mpjpe as calculate_mpjpe
from loss.pose3d import acc_error as calculate_acc_err
from data.const import H36M_JOINT_TO_LABEL, H36M_UPPER_BODY_JOINTS, H36M_LOWER_BODY_JOINTS, H36M_1_DF, H36M_2_DF, \
    H36M_3_DF
from data.reader.h36m import MultipleDataReaderH36M
from data.reader.motion_dataset import MultipleMotionDataset3D
from utils.data import flip_data
from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists, Logger, copy_files
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from utils.learning import load_model, AverageMeter, decay_lr_exponentially
from utils.tools import count_param_numbers
from utils.data import Augmenter2D


def get_root_path():
    return os.path.dirname(__file__)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/h36m/MotionAGFormer-base.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument("--local-rank", type=int, help="Local rank of the process on the node")
    # parser.add_argument('--new-checkpoint', type=str, metavar='PATH', default='checkpoint',
    #                     help='new checkpoint directory')
    # parser.add_argument('--checkpoint-file', type=str, help="checkpoint file name")
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('--num-cpus', default=8, type=int, help='Number of CPU cores')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-name', default=None, type=str)
    parser.add_argument('--wandb-run-id', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    opts = parser.parse_args()
    return opts


def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        return False

    torch.cuda.set_device(args.local_rank)

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"] or os.environ["LOCAL_RANK"], "use pytorch launcher and explicityly state the rank of the process"

    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return True


def distributed_gather_data(result, dist_size, rank):
    result = torch.cat(result)
    if dist_size is not None and len(dist_size) > 1:
        data_dtype = result[0].dtype
        max_len = max(dist_size)
        buffer = [torch.zeros([max_len, *result.shape[1:]], dtype=data_dtype).cuda() for _ in range(len(dist_size))]
        scatter_tensor = torch.zeros_like(buffer[rank], dtype=data_dtype)
        scatter_tensor[:dist_size[rank]] = result[:dist_size[rank]]
        torch.distributed.all_gather(buffer, scatter_tensor)
        result = torch.cat([tensor[:n] for tensor, n in zip(buffer, dist_size)], dim=0)
    result = result.cpu().numpy()
    return result


def train_one_epoch(args, model, train_loader, optimizer, device, losses, is_distributed):
    # some gradients accumulation setups
    accumulate_grads = args.accumulate_grads
    accumulate_iters = args.accumulate_iters if accumulate_grads else 1

    model.train()
    for batch_idx, batch in tqdm(enumerate(train_loader)):
        poses, gt = batch
        batch_size = poses.shape[0]
        poses, gt = poses.to(device), gt.to(device)

        with torch.no_grad():
            if args.root_rel:
                gt = gt - gt[..., 0:1, :]
            else:
                gt[..., 2] = gt[..., 2] - gt[:, 0:1, 0:1, 2]  # Place the depth of first frame root to be 0

        # for efficiency
        tmp_context = model.no_sync if is_distributed and (batch_idx + 1) % accumulate_iters != 0 else nullcontext
        with tmp_context():
            pred = model(poses)

            # optimizer.zero_grad()
            loss_3d_pos = loss_mpjpe(pred, gt)
            loss_3d_scale = n_mpjpe(pred, gt)
            loss_3d_velocity = loss_velocity(pred, gt)
            loss_lv = loss_limb_var(pred)
            loss_lg = loss_limb_gt(pred, gt)
            loss_a = loss_angle(pred, gt)
            loss_av = loss_angle_velocity(pred, gt)

            loss_total = loss_3d_pos + \
                        args.lambda_scale * loss_3d_scale + \
                        args.lambda_3d_velocity * loss_3d_velocity + \
                        args.lambda_lv * loss_lv + \
                        args.lambda_lg * loss_lg + \
                        args.lambda_a * loss_a + \
                        args.lambda_av * loss_av

            losses['3d_pose'].update(loss_3d_pos.item(), batch_size)
            losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
            losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
            losses['lv'].update(loss_lv.item(), batch_size)
            losses['lg'].update(loss_lg.item(), batch_size)
            losses['angle'].update(loss_a.item(), batch_size)
            losses['angle_velocity'].update(loss_av.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)

            # new
            loss_total = loss_total / accumulate_iters

            loss_total.backward()

        # accumulate the gradients
        if ((batch_idx + 1) % accumulate_iters == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()


def evaluate(args, model, test_loader, datareader, device, rank=0, dist_size=None, master=True):
    if master:
        print("[INFO] Evaluation")
    if rank is None:
        rank = 0
    results_all = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            poses, gt = batch
            poses, gt = poses.to(device), gt.to(device)

            if args.flip:
                predicted_3d_pos_1 = model(poses)
                poses_flip = flip_data(poses)
                predicted_3d_pos_flip = model(poses_flip)

                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model(poses)

            if args.root_rel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                gt[:, 0, 0, 2] = 0

            # results_all.append(predicted_3d_pos.cpu().numpy())
            results_all.append(predicted_3d_pos)

    results_all = distributed_gather_data(results_all, dist_size, rank)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    sources = np.array(datareader.dt_dataset['test']['source'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
        
    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]
    factor_clips = factors[split_id_test]

    if args.add_velocity:
        action_clips = action_clips[:, :-1]
        frame_clips = frame_clips[:, :-1]
        gt_clips = gt_clips[:, :-1]
        factor_clips = factor_clips[:, :-1]

    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    jpe_all = np.zeros((num_test_frames, args.num_joints))
    e2_all = np.zeros(num_test_frames)
    acc_err_all = np.zeros(num_test_frames - 2)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    results_joints = [{} for _ in range(args.num_joints)]
    results_accelaration = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
        results_accelaration[action] = []
        for joint_idx in range(args.num_joints):
            results_joints[joint_idx][action] = []

    block_list = ['s_09_act_05_subact_02',
                  's_09_act_10_subact_02',
                  's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        gt = gt_clips[idx]
        pred = results_all[idx]
        
        factor = factor_clips[idx][:, None, None]
        pred *= factor

        # Root-relative Errors
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]
        err1 = calculate_mpjpe(pred, gt)
        jpe = calculate_jpe(pred, gt)
        for joint_idx in range(args.num_joints):
            jpe_all[frame_list, joint_idx] += jpe[:, joint_idx]
        acc_err = calculate_acc_err(pred, gt)
        acc_err_all[frame_list[:-2]] += acc_err
        e1_all[frame_list] += err1
        err2 = calculate_p_mpjpe(pred, gt)
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results_procrustes[action].append(err2)
            acc_err = acc_err_all[idx] / oc[idx]
            results[action].append(err1)
            results_accelaration[action].append(acc_err)
            for joint_idx in range(args.num_joints):
                jpe = jpe_all[idx, joint_idx] / oc[idx]
                results_joints[joint_idx][action].append(jpe)
    
    final_result_joints = [[] for _ in range(args.num_joints)]
    final_result_acceleration = []
    final_result = []
    final_result_procrustes = []
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name'] + action_names
    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
        final_result_acceleration.append(np.mean(results_accelaration[action]))
        for joint_idx in range(args.num_joints):
            final_result_joints[joint_idx].append(np.mean(results_joints[joint_idx][action]))
    summary_table.add_row(['P1'] + final_result)
    summary_table.add_row(['P2'] + final_result_procrustes)
    print(summary_table)
    
    joint_errors = []
    for joint_idx in range(args.num_joints):
        joint_errors.append(
            np.mean(np.array(final_result_joints[joint_idx]))
        )
    joint_errors = np.array(joint_errors)
    e1 = np.mean(np.array(final_result))
    assert round(e1, 4) == round(np.mean(joint_errors), 4), f"MPJPE {e1:.4f} is not equal to mean of joint errors {np.mean(joint_errors):.4f}"
    acceleration_error = np.mean(np.array(final_result_acceleration))
    e2 = np.mean(np.array(final_result_procrustes))
    if master:
        print('Protocol #1 Error (MPJPE):', e1, 'mm')
        print('Acceleration error:', acceleration_error, 'mm/s^2')
        print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
        print('----------')
    return e1, e2, joint_errors, acceleration_error


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe, wandb_id):
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_mpjpe': min_mpjpe,
        'wandb_id': wandb_id,
    }, checkpoint_path)


def train(args, opts):
    if "LOCAL_RANK" in os.environ:
        opts.local_rank = int(os.environ["LOCAL_RANK"])
    is_distributed = init_distributed(opts)
    master = True
    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    else:
        rank = world_size = None

    if is_distributed:
        device = torch.device(opts.local_rank)
    else:
        device = torch.device(0)

    dist_size = None

    train_dataset = MultipleMotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MultipleMotionDataset3D(args, args.subset_list, 'test', rank, world_size)
    datareader = MultipleDataReaderH36M(n_frames=args.n_frames, sample_stride=1,
                                        data_stride_train=args.n_frames // 3, data_stride_test=args.n_frames)
    dist_size = test_dataset.dist_size
    train_dataset.set_detector_type(args.detector_type)
    test_dataset.set_detector_type(args.detector_type)
    
    common_loader_params = {
        'batch_size': args.batch_size,
        'num_workers': opts.num_cpus - 1,
        'pin_memory': True,
        'prefetch_factor': 2,
        'persistent_workers': True
    }
    sampler = DistributedSampler(train_dataset) if is_distributed else None
    if is_distributed:
        train_loader = DataLoader(train_dataset, sampler=sampler, worker_init_fn=worker_init_fn, **common_loader_params)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)

    model = load_model(args)

    if master:
        # create_directory_if_not_exists(opts.new_checkpoint)
        n_params = count_param_numbers(model)
        n_params /= 1000000
        # this get the absolute path of this project
        root_path = get_root_path()
        
        # get currnet running time
        now_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        expr_path = os.path.join(root_path, 'logs', now_time)
        create_directory_if_not_exists(expr_path)
        sys.stdout = Logger(f'{expr_path}/running.log')
        
        print_args(args)
        print(f"[INFO] Number of parameters: {n_params:.2f}M")
        
        # add a summary writer
        ten_writer = SummaryWriter(log_dir=expr_path)
        # copy some files
        copy_files('.', expr_path)
        shutil.copy(os.path.join(root_path, opts.config), expr_path)
        shutil.copytree(os.path.join(root_path, 'loss'), os.path.join(expr_path, 'loss'))
        shutil.copytree(os.path.join(root_path, 'model'), os.path.join(expr_path, 'model'))
        shutil.copytree(os.path.join(root_path, 'utils'), os.path.join(expr_path, 'utils'))

    if torch.cuda.is_available():
        model.to(device)
        if is_distributed:
            model = DistributedDataParallel(model, device_ids=[device], output_device=opts.local_rank)

    lr = args.learning_rate
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr,
                            weight_decay=args.weight_decay)
    
    lr_decay = args.lr_decay
    epoch_start = 0
    min_mpjpe = float('inf')  # Used for storing the best model
    wandb_id = opts.wandb_run_id if opts.wandb_run_id is not None else wandb.util.generate_id()

    if opts.checkpoint:
        # checkpoint_path = os.path.join(opts.checkpoint, opts.checkpoint_file if opts.checkpoint_file else "latest_epoch.pth.tr")
        checkpoint_path = opts.checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            state_dict = {}
            for key in checkpoint['model'].keys():
                state_dict[key.replace('module.', '')] = checkpoint['model'][key].clone()
            ret = model.load_state_dict(state_dict, strict=True)
            print(ret)

            if opts.resume:
                lr = checkpoint['lr']
                epoch_start = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                min_mpjpe = checkpoint['min_mpjpe']
                if 'wandb_id' in checkpoint and opts.wandb_run_id is None:
                    wandb_id = checkpoint['wandb_id']
        else:
            print("[WARN] Checkpoint path is empty. Starting from the beginning")
            opts.resume = False

    if not opts.eval_only:
        if opts.resume:
            if opts.use_wandb and master:
                wandb.init(id=wandb_id,
                        project='MotionMetaFormer',
                        resume="must",
                        settings=wandb.Settings(start_method='fork'))
        else:
            if opts.use_wandb and master:
                print(f"Run ID: {wandb_id}")
                wandb.init(id=wandb_id,
                        name=opts.wandb_name,
                        project='MotionMetaFormer',
                        settings=wandb.Settings(start_method='fork'))
                wandb.config.update({"run_id": wandb_id})
                wandb.config.update(args)
                installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
                wandb.config.update({'installed_packages': installed_packages})

    if master:
        # checkpoint_path_latest = os.path.join(opts.new_checkpoint, 'latest_epoch.pth.tr')
        # checkpoint_path_best = os.path.join(opts.new_checkpoint, 'best_epoch.pth.tr')
        checkpoint_path_latest = os.path.join(expr_path, 'latest_epoch.pth.tr')
        checkpoint_path_best = os.path.join(expr_path, 'best_epoch.pth.tr')

    for epoch in range(epoch_start, args.epochs):
        if is_distributed and sampler is not None:
            sampler.set_epoch(epoch)

        if opts.eval_only:
            evaluate(args, model, test_loader, datareader, device, rank, dist_size, master)
            exit()
        if master:
            print(f"[INFO] epoch {epoch}")
        loss_names = ['3d_pose', '3d_scale', '2d_proj', 'lg', 'lv', '3d_velocity', 'angle', 'angle_velocity', 'total']
        losses = {name: AverageMeter() for name in loss_names}

        train_one_epoch(args, model, train_loader, optimizer, device, losses, is_distributed)

        mpjpe, p_mpjpe, joints_error, acceleration_error = evaluate(args, model, test_loader, datareader, device, rank, dist_size, master)

        if master:
            if mpjpe < min_mpjpe:
                min_mpjpe = mpjpe
                save_checkpoint(checkpoint_path_best, epoch, lr, optimizer, model, min_mpjpe, wandb_id)
            save_checkpoint(checkpoint_path_latest, epoch, lr, optimizer, model, min_mpjpe, wandb_id)

        joint_label_errors = {}
        for joint_idx in range(args.num_joints):
            joint_label_errors[f"eval_joints/{H36M_JOINT_TO_LABEL[joint_idx]}"] = joints_error[joint_idx]

        if master:
            # write some thing
            ten_writer.add_scalar('lr', lr, epoch)
            ten_writer.add_scalar('train/loss_3d_pose', losses['3d_pose'].avg, epoch)
            ten_writer.add_scalar('train/loss_3d_scale', losses['3d_scale'].avg, epoch)
            ten_writer.add_scalar('train/loss_3d_velocity', losses['3d_velocity'].avg, epoch)

            ten_writer.add_scalar('train/loss_2d_proj', losses['2d_proj'].avg, epoch)
            ten_writer.add_scalar('train/loss_lg', losses['lg'].avg, epoch)
            ten_writer.add_scalar('train/loss_lv', losses['lv'].avg, epoch)
            ten_writer.add_scalar('train/loss_angle', losses['angle'].avg, epoch)
            ten_writer.add_scalar('train/angle_velocity', losses['angle_velocity'].avg, epoch)
            ten_writer.add_scalar('train/total', losses['total'].avg, epoch)

            ten_writer.add_scalar('eval/mpjpe', mpjpe, epoch)
            ten_writer.add_scalar('eval/acceleration_error', acceleration_error, epoch)
            ten_writer.add_scalar('eval/min_mpjpe', min_mpjpe, epoch)
            ten_writer.add_scalar('eval/p-mpjpe', p_mpjpe, epoch)
            ten_writer.add_scalar('eval_additional/upper_body_error', np.mean(joints_error[H36M_UPPER_BODY_JOINTS]), epoch)
            ten_writer.add_scalar('eval_additional/lower_body_error', np.mean(joints_error[H36M_LOWER_BODY_JOINTS]), epoch)
            ten_writer.add_scalar('eval_additional/1_DF_error', np.mean(joints_error[H36M_1_DF]), epoch)
            ten_writer.add_scalar('eval_additional/2_DF_error', np.mean(joints_error[H36M_2_DF]), epoch)
            ten_writer.add_scalar('eval_additional/3_DF_error', np.mean(joints_error[H36M_3_DF]), epoch)
            for key in joint_label_errors.keys():
                ten_writer.add_scalar(key, joint_label_errors[key], epoch)

        if opts.use_wandb and master:
            wandb.log({
                'lr': lr,
                'train/loss_3d_pose': losses['3d_pose'].avg,
                'train/loss_3d_scale': losses['3d_scale'].avg,
                'train/loss_3d_velocity': losses['3d_velocity'].avg,
                'train/loss_2d_proj': losses['2d_proj'].avg,
                'train/loss_lg': losses['lg'].avg,
                'train/loss_lv': losses['lv'].avg,
                'train/loss_angle': losses['angle'].avg,
                'train/angle_velocity': losses['angle_velocity'].avg,
                'train/total': losses['total'].avg,
                'eval/mpjpe': mpjpe,
                'eval/acceleration_error': acceleration_error,
                'eval/min_mpjpe': min_mpjpe,
                'eval/p-mpjpe': p_mpjpe,
                'eval_additional/upper_body_error': np.mean(joints_error[H36M_UPPER_BODY_JOINTS]),
                'eval_additional/lower_body_error': np.mean(joints_error[H36M_LOWER_BODY_JOINTS]),
                'eval_additional/1_DF_error': np.mean(joints_error[H36M_1_DF]),
                'eval_additional/2_DF_error': np.mean(joints_error[H36M_2_DF]),
                'eval_additional/3_DF_error': np.mean(joints_error[H36M_3_DF]),
                **joint_label_errors
            }, step=epoch + 1)

        lr = decay_lr_exponentially(lr, lr_decay, optimizer)

    if opts.use_wandb and master:
        artifact = wandb.Artifact(f'model', type='model')
        artifact.add_file(checkpoint_path_latest)
        artifact.add_file(checkpoint_path_best)
        wandb.log_artifact(artifact)


def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)
    args = get_config(opts.config)

    train(args, opts)
    end_time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    print(f"Train ends on {end_time_str}")


if __name__ == '__main__':
    main()
