"""
Using off-the-shelf classifier to guide the diffusion sampling
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
import time
import numpy as np
import csv
import functools
import torchvision.models as models

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    start_time = time.time()
    args = create_argparser().parse_args()
    print('args:', agrs)

    dist_util.setup_dist()

    if dist.get_rank() == 0:
        os.makedirs(args.log_dir, exist_ok=True)

    logger.configure(dir=args.log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    assert args.classifier_type in ['resnet50', 'resnet101']
    if args.classifier_type == 'resnet50':
        resnet_address = './pretrained_models/resnet50_weight_V2.pth'
        resnet = models.resnet50()
    else:
        resnet_address = './pretrained_models/resnet101_weight_V2.pth'
        resnet = models.resnet101()
    for param in resnet.parameters():
        param.required_grad = False
    resnet.load_state_dict(th.load(resnet_address))
    resnet.eval()
    resnet.cuda()

    if (args.softplus_beta < np.inf):
        for name, module in resnet.named_children():
            if isinstance(module, th.nn.ReLU):
                resnet._modules[name] = th.nn.Softplus(beta=args.softplus_beta)
            if name in ['layer1','layer2','layer3','layer4']:
                for sub_name, sub_module in module.named_children():
                    if isinstance(sub_module, models.resnet.Bottleneck):
                        for subsub_name, subsub_module in sub_module.named_children():
                            if isinstance(subsub_module, th.nn.ReLU):
                                resnet._modules[name]._modules[sub_name]._modules[subsub_name] = th.nn.Softplus(beta=args.softplus_beta)
    print('resnet', resnet)
    print('classifier_type', args.classifier_type)
    print('softplus_beta', args.softplus_beta)
    args.classifier_scale = float(args.classifier_scale)
    print('classifier_scale', args.classifier_scale)
    args.joint_temperature = float(args.joint_temperature)
    print('joint_temperature', args.joint_temperature)
    args.margin_temperature_discount = float(args.margin_temperature_discount)
    print('margin_temperature_discount', args.margin_temperature_discount)
    print('time_temperature', args.time_temperature)

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    # use off-the-shelf classifier gradient for guided sampling
    def design_cond_fn(inputs, t, y=None):
        assert y is not None
        with th.enable_grad():
            x = inputs[0]
            pred_xstart = inputs[1]
            pred_xstart = pred_xstart.detach().requires_grad_(True)
            # resnet classifier
            logits = resnet(pred_xstart)
            # temperature
            if args.time_temperature == 'time':
                num_sampling_steps = float(args.timestep_respacing)
                current_time = t[0].item()
                temperature1 = (args.joint_temperature/num_sampling_steps)*(num_sampling_steps-current_time)
                temperature1 = max(temperature1,0.01)
            else:
                temperature1 = args.joint_temperature
            temperature2 = temperature1 * args.margin_temperature_discount

            numerator = th.exp(logits*temperature1)[range(len(logits)), y.view(-1)].unsqueeze(1)
            denominator2 = th.exp(logits*temperature2).sum(1, keepdims=True)
            selected = th.log(numerator / denominator2)
            grads = th.autograd.grad(selected.sum(), pred_xstart)[0] * args.classifier_scale
            return grads

    logger.log("sampling...")
    all_images = []
    all_labels = []
    save_index = 0
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=args.classifier_class, size=(args.batch_size,), device=dist_util.dev()
        )
        if args.fix_class:
            classes = th.ones(size=classes.shape, dtype=int, device=dist_til.dev()) * args.fix_class_index

        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=design_cond_fn,
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

        # save images
        current_num = len(all_images) * args.batch_size
        current_int = int(current_num / 10000)
        if current_int != save_index:
            save_index = current_int
            arr = np.concatenate(all_images, axis=0)
            arr = arr[:current_num]
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[:current_num]
            if dist.get_rank() == 0:
                shape_str = "x".join([str(x) for x in arr.shape])
                out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
                logger.log(f"saving to {out_path}")
                np.savez(out_path, arr, label_arr)

    # save npz file
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        filename = f"samples_{shape_str}.npz"
        out_path = os.path.join(logger.get_dir(), filename)
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    used_time = (time.time() - start_time)
    logger.log(f"sampling complete, used time {used_time}")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        log_dir=None,
        fix_class=False,
        fix_class_index=0,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        classifier_type='resnet101',
        softplus_beta=np.inf,
        joint_temperature=1.0,
        margin_temperature_discount=1.0,
        time_temperature=''
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
