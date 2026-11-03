"""Evaluate a MiniPlaces classifier checkpoint."""
import os
import json
import glob
import argparse

from pprint import pprint
import time

import torch

from cls_libs import load_config, evaluate, build_loader, build_model


def main(args):
    """1. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
        pprint(cfg)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg["dataset"]["test"]) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch >= 0:
            ckpt_file = os.path.join(
                args.ckpt, "epoch_{:03d}.pth.tar".format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, "*.pth.tar")))
            assert len(ckpt_file_list) > 0, "No CKPT file found in the folder!"
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    """2. create dataset / dataloader"""
    device = torch.device(cfg['devices'][0])
    loader = build_loader(cfg, False)

    """3. create model"""
    model = build_model(cfg, pretrained=False).to(device)

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])

    """5. Test the model"""
    start = time.time()
    metrics = evaluate(loader, model, checkpoint.get('epoch', 0), device,
                    eval_aug=args.flip, print_freq=args.print_freq, return_metrics=True)
    metrics["seconds"] = time.time() - start
    metrics["flip_augmentation"] = args.flip
    with open(os.path.join(os.path.dirname(ckpt_file), "metrics.json"), "w") as stream:
        json.dump(metrics, stream, indent=2)
    print('All done! Total time: {:.2f} sec'.format(time.time() - start))
    return metrics["top1"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate an image classifier')
    parser.add_argument('config', help='path to a YAML config')
    parser.add_argument('ckpt', help='checkpoint file or experiment folder')
    parser.add_argument('-e', '--epoch', default=-1, type=int, help='zero-based checkpoint epoch')
    parser.add_argument('-p', '--print-freq', default=10, type=int)
    parser.add_argument('--flip', action='store_true', help='optional horizontal-flip test augmentation')
    main(parser.parse_args())
