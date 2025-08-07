
import os
import json
from collections import Counter

import torch
import numpy as np
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torch.utils.tensorboard import SummaryWriter

from utils.data_loader import DenseCapDataset, DataLoaderPFG
from model.densecap import densecap_resnet50_fpn
from torch.cuda.amp import autocast, GradScaler
from evaluation_runner import quantity_check

torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 12
USE_TB = True
CONFIG_PATH = './model_params'
MODEL_NAME = 'con_densecap_resnet50_fpn_focus_helmet'
IMG_DIR_ROOT = './data/samples'
VG_DATA_PATH = './data/VG-regions-lite.h5'
LOOK_UP_TABLES_PATH = './data/VG-regions-dicts-lite.pkl'
REGIONS_JSON = './data/data/region_descriptions.json'
SPLITS_JSON = './data/data/densecap_splits.json'
MAX_TRAIN_IMAGE = -1
MAX_VAL_IMAGE = -1
NO_BOOST_FACTOR = 15.0


def set_args():
    args = dict(
        backbone_pretrained=True,
        return_features=False,
        feat_size=4096,
        hidden_size=512,
        max_len=16,
        emb_size=512,
        rnn_num_layers=1,
        vocab_size=345,
        fusion_type='init_inject',
        detect_loss_weight=1.0,
        caption_loss_weight=1.0,
        lr=1e-5,
        caption_lr=1e-4,
        weight_decay=0.0,
        batch_size=4,
        use_pretrain_fasterrcnn=True,
        box_detections_per_img=30,
    )
    os.makedirs(CONFIG_PATH, exist_ok=True)
    os.makedirs(os.path.join(CONFIG_PATH, MODEL_NAME), exist_ok=True)
    with open(os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'), 'w') as f:
        json.dump(args, f, indent=2)
    return args


def build_caption_weights():
    regions = json.load(open(REGIONS_JSON, 'r'))
    splits = json.load(open(SPLITS_JSON, 'r'))
    train_ids = set(splits['train'])
    counter = Counter()
    for img in regions:
        if img['id'] in train_ids:
            for r in img['regions']:
                ph = r['phrase'].lower().strip()
                if 'no-helmet' in ph :
                    ph = 'no-helmet'
                elif 'helmet' in ph:
                    ph = 'helmet'
                elif 'no-vest' in ph :
                    ph = 'no-vest'
                elif 'vest' in ph:
                    ph = 'vest'
                counter[ph] += 1

    total = sum(counter.values())
    base = {ph: total / (len(counter) * c) for ph, c in counter.items()}
    for ph in base:
        if 'no-helmet' in ph:
            base[ph] *= NO_BOOST_FACTOR
    return base


def train(args):
    print(f"Model {MODEL_NAME} start training...")

    model = densecap_resnet50_fpn(
        backbone_pretrained=args['backbone_pretrained'],
        feat_size=args['feat_size'],
        hidden_size=args['hidden_size'],
        max_len=args['max_len'],
        emb_size=args['emb_size'],
        rnn_num_layers=args['rnn_num_layers'],
        vocab_size=args['vocab_size'],
        fusion_type=args['fusion_type'],
        box_detections_per_img=args['box_detections_per_img']
    )

    if args['use_pretrain_fasterrcnn']:
        pre = fasterrcnn_resnet50_fpn(pretrained=True)
        model.backbone.load_state_dict(pre.backbone.state_dict(), strict=False)
        model.rpn.load_state_dict(pre.rpn.state_dict(), strict=False)

    model.to(device)
    optimizer = torch.optim.Adam([
        {'params': (p for n, p in model.named_parameters() if p.requires_grad and 'box_describer' not in n)},
        {'params': model.roi_heads.box_describer.parameters(), 'lr': args['caption_lr']}
    ], lr=args['lr'], weight_decay=args['weight_decay'])

    train_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='train')
    val_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='val')
    idx_to_token = train_set.look_up_tables['idx_to_token']

    caption_weights = build_caption_weights()
    sample_weights = []
    for idx in range(len(train_set)):
        _, _, info = train_set[idx]
        phs = info.get('phrases', [])
        if not phs:
            sample_weights.append(1.0)
        else:
            w = max(caption_weights.get(phrase.lower().strip(), 1.0) for phrase in phs)
            sample_weights.append(w)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    if MAX_TRAIN_IMAGE > 0:
        train_set = Subset(train_set, range(MAX_TRAIN_IMAGE))
    if MAX_VAL_IMAGE > 0:
        val_set = Subset(val_set, range(MAX_VAL_IMAGE))

    train_loader = DataLoaderPFG(
        train_set,
        batch_size=args['batch_size'],
        shuffle=False,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=DenseCapDataset.collate_fn
    )

    scaler = GradScaler()
    writer = SummaryWriter() if USE_TB else None
    iter_counter = 0
    best_map = 0.0

    for epoch in range(MAX_EPOCHS):
        for batch_idx, (imgs, targets, _) in enumerate(train_loader):
            model.train()
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with autocast():
                try:
                    losses = model(imgs, targets)

                    # Check if any of the loss values are NaN or zeroed-out (no proposals)
                    if not losses or any(torch.isnan(v) or v.item() == 0 for v in losses.values()):
                        print(f"⚠️ Skipping batch {batch_idx}: empty or invalid loss.")
                        continue

                except Exception as e:
                    print(f"❌ Exception during model forward pass on batch {batch_idx}: {e}")
                    continue

            detect_loss = (
                losses['loss_objectness']
                + losses['loss_rpn_box_reg']
                + losses['loss_classifier']
                + losses['loss_box_reg']
            )

            caption_loss = losses['loss_caption']

            total_loss = (
                args['detect_loss_weight'] * detect_loss
                + args['caption_loss_weight'] * caption_loss
            )

            if USE_TB:
                writer.add_scalar('loss/total', total_loss.item(), iter_counter)
                writer.add_scalar('loss/detect', detect_loss.item(), iter_counter)
                writer.add_scalar('loss/caption', caption_loss.item(), iter_counter)

            if iter_counter % (len(train_set) // (args['batch_size'] * 16)) == 0:
                print(f"[{epoch}][{batch_idx}] total_loss {total_loss.item():.3f}")
                for k, v in losses.items():
                    print(f"  <{k}> {v:.3f}")

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if iter_counter > 0 and iter_counter % len(train_set) == 0:
                results = quantity_check(model, val_set, idx_to_token, device, max_iter=-1, verbose=True)
                if results['map'] > best_map:
                    best_map = results['map']
                    save_path = os.path.join('model_params', f"{MODEL_NAME}_{iter_counter}.pth.tar")
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'results_on_val': results, 'iterations': iter_counter}, save_path)
                if USE_TB:
                    writer.add_scalar('metric/map', results['map'], iter_counter)
                    writer.add_scalar('metric/det_map', results['detmap'], iter_counter)

            iter_counter += 1

    end_path = os.path.join('model_params', f"{MODEL_NAME}_end.pth.tar")
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'results_on_val': results, 'iterations': iter_counter}, end_path)

    if USE_TB:
        writer.close()


if __name__ == '__main__':
    args = set_args()
    train(args)