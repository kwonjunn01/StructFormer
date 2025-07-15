import os
import time
import warnings
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torchvision.utils import save_image
from scipy.stats import mode  # 대표 mask 계산용

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter, NativeScaler

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_checkpoint_only, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from PIL import Image, ImageDraw

import json
from angular_dispersion import compute_angular_dispersion, compute_angular_dispersion_per_token, plot_angular_dispersions, save_dispersions_to_file

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

warnings.filterwarnings("ignore", module="PIL")

def parse_option():
    parser = argparse.ArgumentParser('StructFormer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-set', type=str, default='imagenet', help='dataset to use')
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='native', choices=['native', 'O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default='debug', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--num_workers', type=int, default=8, help="")
    parser.add_argument('--vis', action='store_true', help="visualization")
    # parser.add_argument('--warmup_epochs', type=int, default=20, help="#epoches for warm up")
    # parser.add_argument('--epochs', type=int, default=300, help="#epoches")
    # parser.add_argument('--lr', type=float, default=5e-4, help="max learning rate for training")
    # parser.add_argument('--min_lr', type=float, default=5e-6, help="min learning rate for training")
    # parser.add_argument('--warmup_lr', type=float, default=5e-7, help="learning rate to start warmup")
    # parser.add_argument('--weight_decay', type=float, default=5e-2, help="l2 reguralization")

    # local rank is obtained using os.environ in newr version
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    parser.add_argument("--img_size", type=int, default=224, help='input resolution for image')
    parser.add_argument("--embed_dim", type=int, nargs='+', default=None, help='size of embedding')
    parser.add_argument("--impl_type", type=str, default='', help='options to use for different methods')

    # arguments relevant to our experiment
    parser.add_argument('--group_type', type=str, default='constant', help='group size type')
    parser.add_argument('--use_cpe', action='store_true', help='whether to use conditional positional encodings')
    parser.add_argument('--pad_type', type=int, default=0, help='0 to pad in one direction, otherwise 1')
    parser.add_argument('--no_mask', action='store_true', help='whether to use mask after padding')
    parser.add_argument('--adaptive_interval', action='store_true', help='interval change with the group size')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(args, config):
    # create token_label dataset
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, args)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0" and config.AMP_OPT_LEVEL != "native":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module
    loss_scaler = NativeScaler()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint_only(config, model_without_ddp, optimizer, lr_scheduler, logger)
        validate_token_assignment(data_loader_val, model)
        if config.VISUALIZATION:
            
            save_semantic_masks(data_loader_val, model)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        
        # Measure angular dispersion
        logger.info("Starting angular dispersion analysis...")
        measure_angular_dispersion(config, data_loader_val, model)
        
        if config.EVAL_MODE:
            return

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.WEIGHT_OUTPUT)
        if resume_file:
            if config.MODEL.RESUME and os.path.getmtime(config.MODEL.RESUME) >= os.path.getmtime(resume_file):
                pass
            else:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
                config.defrost()
                config.MODEL.RESUME = resume_file
                config.freeze()
                logger.info(f'auto resuming from {resume_file}')
                max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
                acc1, acc5, loss = validate(config, data_loader_val, model)
                logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
                if config.EVAL_MODE:
                    return
        else:
            logger.info(f'no checkpoint found in {config.WEIGHT_OUTPUT}, ignoring auto resume')

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return
    # if config.VISUALIZATION:
    #     print("visual!!!!!!!!!!!!")
    #     save_stage_representative_masks(data_loader_val, model)
    #     return
    if config.MODEL.FROM_PRETRAIN:
        config.defrost()
        config.MODEL.RESUME = config.MODEL.FROM_PRETRAIN
        config.freeze()
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler)
        if dist.get_rank() == 0 and epoch == config.TRAIN.EPOCHS - 1:
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)
        if config.DATA.DATASET != "ImageNet22K" or epoch % 10 == 0 or epoch == config.TRAIN.EPOCHS - 1:
            acc1, acc5, loss = validate(config, data_loader_val, model, epoch)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            if dist.get_rank() == 0 and acc1 >= max_accuracy: ## save best
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, best=True)
            if dist.get_rank() == 0:
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, last=True)
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Epoch: {epoch:d}, Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=(config.AMP_OPT_LEVEL=="native")):
            outputs, sim_token, kl_loss = model(samples)
            loss = criterion(outputs, targets)
            loss = loss + 0.1*criterion(sim_token, targets)
            loss = loss + kl_loss*0.001
            if config.TRAIN.ACCUMULATION_STEPS > 1:
                loss = loss / config.TRAIN.ACCUMULATION_STEPS
                if config.AMP_OPT_LEVEL != "O0" and config.AMP_OPT_LEVEL != "native":
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(amp.master_params(optimizer))
                else:
                    loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(model.parameters())
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                optimizer.zero_grad()
                if config.AMP_OPT_LEVEL != "O0" and config.AMP_OPT_LEVEL != "native":
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(amp.master_params(optimizer))
                    optimizer.step()
                elif config.AMP_OPT_LEVEL == "native":
                    loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD, parameters=model.parameters())
                    grad_norm = 0
                    for p in model.parameters():
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                    grad_norm = grad_norm ** 0.5
                else:
                    loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(model.parameters())
                    optimizer.step()
                lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), samples.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 or idx == len(data_loader) - 1:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}], '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}, '
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f}), '
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}), '
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f}), '
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, epoch=0):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=(config.AMP_OPT_LEVEL=="native")):
            output,_, kl_loss = model(images)

            # measure accuracy and record loss
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 or idx == len(data_loader) - 1:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Epoch {epoch:d}\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

@torch.no_grad()
def measure_angular_dispersion(config, data_loader, model, save_dir='angular_dispersion_results'):
    """
    Measure angular dispersion between patch embeddings and semantic tokens at each layer.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    all_dispersions = {}
    num_samples = 0
    
    logger.info("Measuring angular dispersion across layers...")
    
    for idx, (images, target) in enumerate(data_loader):
        if idx >= 10:  # Limit to first 10 batches for efficiency
            break
            
        images = images.cuda(non_blocking=True)
        
        # Get layer outputs
        with torch.cuda.amp.autocast(enabled=(config.AMP_OPT_LEVEL=="native")):
            _, _, _, layer_outputs = model(images, return_layer_outputs=True)
        
        # Process each layer's output
        for layer_info in layer_outputs:
            layer_idx = layer_info['layer_idx']
            patch_embeddings = layer_info['patch_embeddings']  # (B, N, D)
            semantic_tokens = layer_info['semantic_tokens']   # (B, K, D)
            
            # Compute overall dispersion
            overall_dispersion = compute_angular_dispersion(patch_embeddings, semantic_tokens)
            
            # Compute per-token dispersion
            per_token_dispersions = compute_angular_dispersion_per_token(patch_embeddings, semantic_tokens)
            
            # Store results
            layer_key = f'layer_{layer_idx}'
            if layer_key not in all_dispersions:
                all_dispersions[layer_key] = [[], []]  # [overall, per_token]
            
            all_dispersions[layer_key][0].append(overall_dispersion)
            all_dispersions[layer_key][1].append(per_token_dispersions)
        
        num_samples += 1
        
        if idx % 5 == 0:
            logger.info(f'Processed batch [{idx+1}/10]')
    
    # Average dispersions across samples
    avg_dispersions = {}
    for layer_key in all_dispersions:
        overall_list = all_dispersions[layer_key][0]
        per_token_list = all_dispersions[layer_key][1]
        
        # Average overall dispersion
        avg_overall = np.mean(overall_list)
        
        # Average per-token dispersions
        num_tokens = len(per_token_list[0])
        avg_per_token = []
        for token_idx in range(num_tokens):
            token_dispersions = [per_token_list[i][token_idx] for i in range(len(per_token_list))]
            avg_per_token.append(np.mean(token_dispersions))
        
        avg_dispersions[layer_key] = [avg_overall] + avg_per_token
    
    # Save results
    plot_path = os.path.join(save_dir, 'angular_dispersion_plot.png')
    text_path = os.path.join(save_dir, 'angular_dispersion_values.txt')
    
    plot_angular_dispersions(avg_dispersions, plot_path)
    save_dispersions_to_file(avg_dispersions, text_path)
    
    logger.info(f"Angular dispersion analysis completed. Results saved to {save_dir}")
    
    return avg_dispersions

def compute_assignment_consistency(p_mask, c_mask, mapping={0: [0, 1], 1: [2, 3], 2: [4, 5]}):
    """
    p_mask: numpy array, shape (N,), parent's token assignments (정수 값)
    c_mask: numpy array, shape (N,), child's token assignments (정수 값)
    mapping: dict, 부모 값 -> list of 허용되는 자식 값들.
    
    반환: 할당 일관성 (0~1 사이 값, 각 patch마다 자식 값이 mapping에 속하면 맞는 것으로 간주)
    """
    p_flat = p_mask.flatten()
    c_flat = c_mask.flatten()
    total = p_flat.size
    correct = 0
    for p_val, c_val in zip(p_flat, c_flat):
        if c_val in mapping.get(p_val, []):
            correct += 1
    return correct / total

def validate_token_assignment(dataloader, model):
    """
    전체 validation 동안, Stage 3와 (존재하면) Stage 4의 각 block에서,
    각 head별로 부모 토큰(mask)과 자식 토큰(mask)의 할당 일관성을 측정합니다.
    
    각 block의 p_mask (또는 p_mask_)와 semantic_mask (또는 semantic_mask_)가
    주어질 때, 부모 mask 값이 0이면 자식은 0, 부모가 1이면 자식은 [2,3], 부모가 2이면 자식은 [4,5]여야 한다고 가정합니다.
    
    반환:
        avg_metrics: dict, key: (stage, block_idx, head), value: 해당 block, head의 평균 일관성.
    """
    model.eval()
    metrics = {}  # key: (stage, block_idx, head) -> list of per-batch consistency
    mapping = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    with torch.no_grad():
        for images, _ in dataloader:
            _ = model(images)
            model_to_use = model.module if hasattr(model, "module") else model
            
            # Stage 3: p_mask_ and semantic_mask_
            if len(model_to_use.layers) > 2:
                stage3 = model_to_use.layers[2]
                for block_idx, block in enumerate(stage3.blocks):
                    if (hasattr(block.attn, 'p_mask') and block.attn.p_mask is not None and
                        hasattr(block.attn, 'semantic_mask') and block.attn.semantic_mask is not None):
                        p_mask = block.attn.p_mask.detach().cpu().numpy()  # shape: [B, num_heads, N]
                        c_mask = block.attn.semantic_mask.detach().cpu().numpy()
                        B, num_heads, N = p_mask.shape
                        head_consistencies = []
                        for head in range(num_heads):
                            p_mask_head = p_mask[0, head].reshape(-1)
                            c_mask_head = c_mask[0, head].reshape(-1)
                            consistency = compute_assignment_consistency(p_mask_head, c_mask_head, mapping)
                        
                            
                            head_consistencies.append(consistency)
                        key = f"stage3_block{block_idx}_head{head}"
                        avg_consistency = np.mean(head_consistencies)
                        metrics.setdefault(key, []).append(avg_consistency)
                    else:
                        print(f"Stage 3 Block {block_idx}: p_mask_ or semantic_mask_ not computed.")
            
            # Stage 4: (if exists) using p_mask and semantic_mask
            if len(model_to_use.layers) > 3:
                stage4 = model_to_use.layers[3]
                for block_idx, block in enumerate(stage4.blocks):
                    if (hasattr(block.attn, 'p_mask') and block.attn.p_mask is not None and
                        hasattr(block.attn, 'semantic_mask') and block.attn.semantic_mask is not None):
                        p_mask = block.attn.p_mask.detach().cpu().numpy()
                        c_mask = block.attn.semantic_mask.detach().cpu().numpy()
                        B, num_heads, N = p_mask.shape
                        head_consistencies = []
                        for head in range(num_heads):
                            p_mask_head = p_mask[0, head].reshape(-1)
                            c_mask_head = c_mask[0, head].reshape(-1)
                            consistency = compute_assignment_consistency(p_mask_head, c_mask_head, mapping)
                            head_consistencies.append(consistency)
                        avg_consistency = np.mean(head_consistencies)
                        key = f"stage4_block{block_idx}_head{head}"
                        metrics.setdefault(key, []).append(avg_consistency)
                    else:
                        print(f"Stage 4 Block {block_idx}: p_mask or semantic_mask not computed.")
    
    
    
    # 평균 일관성 계산
    avg_metrics = {key: np.mean(vals) for key, vals in metrics.items()}
    metrics_save_path = "./res.json"
    with open(metrics_save_path, 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    print(f"Saved assignment metrics to {metrics_save_path}")
    return avg_metrics
@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return
    


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image
from scipy.stats import mode

# ----- 대표 마스크 계산 함수 (keepdims 적용) -----
def compute_representative_mask(masks):
    """
    masks: numpy array, shape (n, H, W) – n개의 마스크가 들어있는 배열.
    각 픽셀에 대해 n개 중 최빈값(mode)을 계산하여 대표 mask를 만듭니다.
    반환: (H, W) numpy array.
    """
    res = mode(masks, axis=0, keepdims=True).mode  # 결과 shape: (1, H, W)
    return np.squeeze(res, axis=0)

def upscale_and_save(image_array, file_path, scale_factor=10):
    """
    image_array: (H, W, 3) numpy 배열 (uint8) – 저장할 이미지.
    file_path: 저장할 파일 경로.
    scale_factor: 업샘플 배율 (기본 10).
    """
    img_pil = Image.fromarray(image_array)
    new_size = (img_pil.width * scale_factor, img_pil.height * scale_factor)
    img_upscaled = img_pil.resize(new_size, resample=Image.NEAREST)
    img_upscaled.save(file_path)
    print(f"Saved upscaled image: {file_path}")

def colorize_mask(mask, palette):
    """
    mask: 2D numpy array (H x W) – 정수 mask.
    palette: dict, 각 key에 대해 [R, G, B] 값 (0~255).
    
    반환: (H x W x 3) uint8 배열 – palette에 따라 색칠된 이미지.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for key, color in palette.items():
        color_mask[mask == key] = color
    return color_mask

def denormalize(image_tensor, mean, std):
    """
    image_tensor: (C, H, W) 텐서, Normalize가 적용된 이미지.
    mean, std: 각 채널의 평균과 표준편차.
    
    반환: denormalized된 이미지 텐서.
    """
    if image_tensor.device.type != "cpu":
        image_tensor = image_tensor.cpu()
    image_tensor = image_tensor.clone()
    for t, m, s in zip(image_tensor, mean, std):
        t.mul_(s).add_(m)
    return image_tensor

def resize_mask(mask, target_shape):
    """
    mask: numpy array, shape (H, W) – 정수 mask.
    target_shape: (H_target, W_target)
    
    반환: target_shape으로 리사이즈된 mask (최근접 이웃 방식).
    """
    img = Image.fromarray(mask.astype(np.uint8), mode="L")
    img_resized = img.resize((target_shape[1], target_shape[0]), resample=Image.NEAREST)
    return np.array(img_resized)
def draw_dotted_grid(pil_img, grid_size, line_color=(255,255,255), line_width=1, dot_length=2, gap=2):
    """
    pil_img: PIL Image 객체.
    grid_size: 각 cell의 크기 (픽셀 단위). (예: 32)
    line_color: 점선의 색상.
    line_width: 선 두께.
    dot_length: 각 점선 segment의 길이.
    gap: 점선 segment 사이의 간격.
    반환: grid가 그려진 PIL Image 객체.
    """
    draw = ImageDraw.Draw(pil_img)
    width, height = pil_img.size
    # 세로 방향 점선 그리기 (vertical)
    for x in range(grid_size, width, grid_size):
        y = 0
        while y < height:
            y_end = min(y + dot_length, height)
            draw.line([(x, y), (x, y_end)], fill=line_color, width=line_width)
            y += dot_length + gap
    # 가로 방향 점선 그리기 (horizontal)
    for y in range(grid_size, height, grid_size):
        x = 0
        while x < width:
            x_end = min(x + dot_length, width)
            draw.line([(x, y), (x_end, y)], fill=line_color, width=line_width)
            x += dot_length + gap
    return pil_img
def resize_color_image(color_img, target_shape):
    """
    color_img: (H, W, 3) numpy 배열, uint8 – 색칠된 mask 이미지.
    target_shape: (target_H, target_W)
    
    반환: target_shape으로 리사이즈된 color_img (최근접 이웃 방식).
    """
    img = Image.fromarray(color_img)
    img_resized = img.resize((target_shape[1], target_shape[0]), resample=Image.NEAREST)
    return np.array(img_resized)

def overlay_mask_on_image(input_img, mask_img, alpha=0.5):
    """
    input_img: (H, W, 3) numpy 배열, uint8 – 원본 이미지.
    mask_img: (H, W, 3) numpy 배열, uint8 – 색칠된 mask.
    alpha: mask의 투명도 (0~1).
    
    반환: 오버레이된 이미지 (H, W, 3) uint8.
    """
    input_float = input_img.astype(np.float32)
    mask_float = mask_img.astype(np.float32)
    overlay = np.clip((1 - alpha) * input_float + alpha * mask_float, 0, 255).astype(np.uint8)
    return overlay

# ----- 색상 팔레트 -----
# 부모 대표 마스크용 (Stage 1,2 및 Stage 3의 부모) → 순수한 빨강, 초록, 파랑
parent_palette = {
    0: [255, 140, 0],    # orange
    1: [0, 100, 0],    # 초록
    2: [139, 0, 139]     # 파랑
}
# Stage 3의 자식 대표 마스크용 → 빨강, 연빨, 파랑, 연파, 초록, 연초
child_palette = {
    0: [255, 140, 0],       # 빨강
    1: [255, 255, 0],   # 연빨
    2: [0, 100, 0],       # 초록
    3: [152, 251, 152],    # 연초
    4: [139, 0, 139],       # 파랑
    5: [255, 20, 147],   # 연파
}

# ----- Stage별 대표 마스크 통합 및 저장 코드 -----

def save_semantic_masks(dataloader, model, save_dir="./semantic_masks", scale_factor=10, overlay_alpha=0.5):
    """
    StructFormer 모델의 forward 수행 후, 각 배치마다 입력 이미지와 함께  
    Stage 2와 Stage 3의 대표 마스크(부모 및 자식)를 저장합니다.
    
    - Stage 0,1은 single token이므로 생략합니다.
    - Stage 2와 Stage 3는 double token이므로, 각 블록별로 대표 마스크를 계산합니다.
    - 계산된 마스크는 원본 입력 이미지 크기로 확장한 후,  
      각 grid cell 경계에 얇은 점선(예: 흰색 점선)을 추가하여 오버레이합니다.
    
    부모 마스크: 반올림 후 0~2 clip → parent_palette,
    자식 마스크: 반올림 후 0~5 clip → child_palette
    
    Args:
        dataloader: (images, labels) 튜플 반환 DataLoader.
        model: 수정된 StructFormer 모델 (DDP 사용 시 model.module).
        save_dir: 저장 폴더 (기본 "./semantic_masks").
        scale_factor: 업샘플 배율 (최종 저장 시 적용).
        overlay_alpha: 오버레이 시 mask 투명도 (0~1, 기본 0.5).
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    
    for idx, (images, _) in enumerate(dataloader):
        if rank != 0:
            with torch.no_grad():
                _ = model(images)
            continue
        
        batch_folder = os.path.join(save_dir, str(idx))
        os.makedirs(batch_folder, exist_ok=True)
        
        with torch.no_grad():
            _ = model(images)
        
        # 입력 이미지 저장 (denormalize 적용)
        input_image = images[0].cpu()
        input_image_denorm = denormalize(input_image.clone(), [0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        image_save_path = os.path.join(batch_folder, "input_image.png")
        save_image(input_image_denorm, image_save_path)
        print(f"Saved input image: {image_save_path}")
        
        # 원본 입력 이미지를 numpy 배열로 변환 (HxWx3)
        input_np = input_image_denorm.mul(255).clamp(0,255).permute(1,2,0).cpu().numpy().astype(np.uint8)
        
        model_to_use = model.module if hasattr(model, "module") else model
        if len(model_to_use.layers) > 1:
            stage1 = model_to_use.layers[1]
            for block_idx, block in enumerate(stage1.blocks):
                # Stage 1은 double token이 아니므로 parent token만 있다고 가정
                if hasattr(block.attn, 'semantic_mask') and block.attn.semantic_mask is not None:
                    p_mask = block.attn.semantic_mask  # [B, num_heads, N]
                    B, num_heads, N = p_mask.shape
                    G = int(np.sqrt(N))
                    if G * G != N:
                        print(f"Stage 1 Block {block_idx}: N({N}) is not a perfect square.")
                        continue
                    rep_p_masks = []
                    for head in range(num_heads):
                        parent_mask_channel = p_mask[0, head].cpu().numpy().astype(np.int32).reshape(G, G)
                        rep_p_masks.append(parent_mask_channel)
                    
                    rep_parent = compute_representative_mask(np.array(rep_p_masks))
                    rep_parent = rep_parent.reshape(G, G)
                    rep_parent_discrete = np.clip(np.round(rep_parent).astype(np.int32), 0, 2)
                    rep_parent_color = colorize_mask(rep_parent_discrete, parent_palette)
                    rep_parent_color_up = resize_color_image(rep_parent_color, input_np.shape[:2])
                    grid_size = int(round(input_np.shape[0] / G))
                    pil_mask = Image.fromarray(rep_parent_color_up)
                    pil_mask = draw_dotted_grid(pil_mask, grid_size=grid_size, line_color=(255,255,255),
                                                 line_width=1, dot_length=2, gap=2)
                    rep_parent_color_up_grid = np.array(pil_mask)
                    overlay_parent = overlay_mask_on_image(input_np, rep_parent_color_up_grid, alpha=overlay_alpha)
                    rep_parent_file = os.path.join(batch_folder, f"rep_parent_stage_1_block_{block_idx}.png")
                    upscale_and_save(overlay_parent, rep_parent_file, scale_factor=1)
                    print(f"Saved overlaid rep parent mask for Stage 1 block {block_idx}: {rep_parent_file}")
                else:
                    print(f"Stage 1 Block {block_idx}: p_mask not computed.")
        # --- Stage 2 처리: 각 블록별로 저장 ---
        if len(model_to_use.layers) > 2:
            stage2 = model_to_use.layers[2]
            for block_idx, block in enumerate(stage2.blocks):
                if (hasattr(block.attn, 'p_mask') and block.attn.p_mask is not None and
                    hasattr(block.attn, 'semantic_mask') and block.attn.semantic_mask is not None):
                    
                    p_mask = block.attn.p_mask   # [B, num_heads, N]
                    sem_mask = block.attn.semantic_mask  # [B, num_heads, N]
                    B, num_heads, N = p_mask.shape
                    G = int(np.sqrt(N))
                    if G * G != N:
                        print(f"Stage 2 Block {block_idx}: N({N}) is not a perfect square.")
                        continue
                    
                    rep_p_masks = []
                    rep_child_masks = []
                    for head in range(num_heads):
                        parent_mask_channel = p_mask[0, head].cpu().numpy().astype(np.int32).reshape(G, G)
                        rep_p_masks.append(parent_mask_channel)
                        child_mask_channel = sem_mask[0, head].cpu().numpy().astype(np.int32).reshape(G, G)
                        rep_child_masks.append(child_mask_channel)
                    
                    # 부모 대표 mask 계산 및 quantization (0~2)
                    rep_parent = compute_representative_mask(np.array(rep_p_masks))
                    rep_parent = rep_parent.reshape(G, G)
                    rep_parent_discrete = np.clip(np.round(rep_parent).astype(np.int32), 0, 2)
                    rep_parent_color = colorize_mask(rep_parent_discrete, parent_palette)
                    # 확장: 현재 rep_parent_color의 크기 (GxG)를 원본 입력 이미지 크기로 확장
                    rep_parent_color_up = resize_color_image(rep_parent_color, input_np.shape[:2])
                    # 그 위에 grid를 그립니다. grid cell 크기는 input_np_height / G
                    grid_size = int(round(input_np.shape[0] / G))
                    pil_mask = Image.fromarray(rep_parent_color_up)
                    pil_mask = draw_dotted_grid(pil_mask, grid_size=grid_size, line_color=(255,255,255), line_width=1, dot_length=2, gap=2)
                    rep_parent_color_up_grid = np.array(pil_mask)
                    overlay_parent = overlay_mask_on_image(input_np, rep_parent_color_up_grid, alpha=overlay_alpha)
                    rep_parent_file = os.path.join(batch_folder, f"rep_parent_stage_2_block_{block_idx}.png")
                    upscale_and_save(overlay_parent, rep_parent_file, scale_factor=1)
                    print(f"Saved overlaid rep parent mask for Stage 2 block {block_idx}: {rep_parent_file}")
                    
                    # 자식 대표 mask 계산 및 quantization (0~5)
                    rep_child = compute_representative_mask(np.array(rep_child_masks))
                    rep_child = rep_child.reshape(G, G)
                    rep_child_discrete = np.clip(np.round(rep_child).astype(np.int32), 0, 5)
                    rep_child_color = colorize_mask(rep_child_discrete, child_palette)
                    rep_child_color_up = resize_color_image(rep_child_color, input_np.shape[:2])
                    pil_mask_child = Image.fromarray(rep_child_color_up)
                    pil_mask_child = draw_dotted_grid(pil_mask_child, grid_size=grid_size, line_color=(255,255,255), line_width=1, dot_length=2, gap=2)
                    rep_child_color_up_grid = np.array(pil_mask_child)
                    overlay_child = overlay_mask_on_image(input_np, rep_child_color_up_grid, alpha=overlay_alpha)
                    rep_child_file = os.path.join(batch_folder, f"rep_child_stage_2_block_{block_idx}.png")
                    upscale_and_save(overlay_child, rep_child_file, scale_factor=1)
                    print(f"Saved overlaid rep child mask for Stage 2 block {block_idx}: {rep_child_file}")
                else:
                    print(f"Stage 2 Block {block_idx}: p_mask or semantic_mask not computed.")
        else:
            print("Model does not have Stage 2.")
        
        # --- Stage 3 처리: 각 블록별로 저장 ---
        if len(model_to_use.layers) > 3:
            stage3 = model_to_use.layers[3]
            for block_idx, block in enumerate(stage3.blocks):
                if (hasattr(block.attn, 'p_mask') and block.attn.p_mask is not None and
                    hasattr(block.attn, 'semantic_mask') and block.attn.semantic_mask is not None):
                    
                    p_mask = block.attn.p_mask_   # [B, num_heads, N]
                    sem_mask = block.attn.semantic_mask_  # [B, num_heads, N]
                    B, num_heads, N = p_mask.shape
                    G = int(np.sqrt(N))
                    if G * G != N:
                        print(f"Stage 3 Block {block_idx}: N({N}) is not a perfect square.")
                        continue
                    
                    rep_p_masks = []
                    rep_child_masks = []
                    for head in range(num_heads):
                        parent_mask_channel = p_mask[0, head].cpu().numpy().astype(np.int32).reshape(G, G)
                        rep_p_masks.append(parent_mask_channel)
                        child_mask_channel = sem_mask[0, head].cpu().numpy().astype(np.int32).reshape(G, G)
                        rep_child_masks.append(child_mask_channel)
                    
                    # 부모 대표 mask (Stage 3, 각 블록)
                    rep_parent = compute_representative_mask(np.array(rep_p_masks))
                    rep_parent = rep_parent.reshape(G, G)
                    rep_parent_discrete = np.clip(np.round(rep_parent).astype(np.int32), 0, 2)
                    rep_parent_color = colorize_mask(rep_parent_discrete, parent_palette)
                    rep_parent_color_up = resize_color_image(rep_parent_color, input_np.shape[:2])
                    grid_size = int(round(input_np.shape[0] / G))
                    pil_mask = Image.fromarray(rep_parent_color_up)
                    pil_mask = draw_dotted_grid(pil_mask, grid_size=grid_size, line_color=(255,255,255), line_width=1, dot_length=2, gap=2)
                    rep_parent_color_up_grid = np.array(pil_mask)
                    overlay_parent = overlay_mask_on_image(input_np, rep_parent_color_up_grid, alpha=overlay_alpha)
                    rep_parent_file = os.path.join(batch_folder, f"rep_parent_stage_3_block_{block_idx}.png")
                    upscale_and_save(overlay_parent, rep_parent_file, scale_factor=1)
                    print(f"Saved overlaid rep parent mask for Stage 3 block {block_idx}: {rep_parent_file}")
                    
                    # 자식 대표 mask (Stage 3, 각 블록)
                    rep_child = compute_representative_mask(np.array(rep_child_masks))
                    rep_child = rep_child.reshape(G, G)
                    rep_child_discrete = np.clip(np.round(rep_child).astype(np.int32), 0, 5)
                    rep_child_color = colorize_mask(rep_child_discrete, child_palette)
                    rep_child_color_up = resize_color_image(rep_child_color, input_np.shape[:2])
                    pil_mask_child = Image.fromarray(rep_child_color_up)
                    pil_mask_child = draw_dotted_grid(pil_mask_child, grid_size=grid_size, line_color=(255,255,255), line_width=1, dot_length=2, gap=2)
                    rep_child_color_up_grid = np.array(pil_mask_child)
                    overlay_child = overlay_mask_on_image(input_np, rep_child_color_up_grid, alpha=overlay_alpha)
                    rep_child_file = os.path.join(batch_folder, f"rep_child_stage_3_block_{block_idx}.png")
                    upscale_and_save(overlay_child, rep_child_file, scale_factor=1)
                    print(f"Saved overlaid rep child mask for Stage 3 block {block_idx}: {rep_child_file}")
                else:
                    print(f"Stage 3 Block {block_idx}: p_mask or semantic_mask not computed.")
        else:
            print("Model does not have Stage 3.")







if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0" and config.AMP_OPT_LEVEL != "native":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.LOG_OUTPUT,    exist_ok=True)
    os.makedirs(config.WEIGHT_OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.LOG_OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.LOG_OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(args, config)
