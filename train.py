import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import datetime
import timm
import torch
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from network_files import MaskRCNN, BackboneWithFPN
from my_dataset_coco import CocoDetection
from train_utils import train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups



def create_model(num_classes, load_pretrain_weights=True):


    backbone = timm.create_model('resnext50_32x4d',
                                 pretrained=load_pretrain_weights,
                                 features_only=True)
    return_layers = {"layer1": "0",
                     "layer2": "1",
                     "layer3": "2",
                     "layer4": "3"}
    in_channels_list = [256, 512, 1024, 2048]
    new_backbone = create_feature_extractor(
        backbone,
        return_nodes=return_layers
    )
    backbone = BackboneWithFPN(
        new_backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),
        re_getter=False
    )

    model = MaskRCNN(backbone, num_classes=num_classes)

    if load_pretrain_weights:
        weights_dict = torch.load("backbone/resnext50_fpn.pth", map_location="cpu")
        for k in list(weights_dict.keys()):
           if ("box_predictor" in k) or ("mask_fcn_logits" in k):
               del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = f"det_results{now}.txt"
    seg_results_file = f"seg_results{now}.txt"
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    data_root = args.data_path

    train_dataset = CocoDetection(data_root, "train", data_transform["train"])
    train_sampler = None

    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % nw)

    if train_sampler:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    val_dataset = CocoDetection(data_root, "val", data_transform["val"])
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn)
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain)

    model.to(device)

    train_loss = []
    learning_rate = []
    val_map = []

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=args.lr_steps,
                                                              eta_min=0.0001)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        lr_scheduler.step()

        det_info, seg_info = utils.evaluate(model, val_data_loader, device=device)

        with open(det_results_file, "a") as f:
            result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")
        val_map.append(det_info[1])

        with open(seg_results_file, "a") as f:
            result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(det_info[1])


    save_files = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch}
    if args.amp:
        save_files["scaler"] = scaler.state_dict()
    torch.save(save_files, "./save_weights/model_{}.pth".format(epoch))

    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--data-path', default='data/coco2017', help='dataset')
    parser.add_argument('--num-classes', default=2, type=int, help='num_classes')
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')

    parser.add_argument('--lr-steps', default=50, nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--pretrain", type=bool, default=True, help="load COCO pretrain weights.")
    parser.add_argument("--amp", default=True, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
