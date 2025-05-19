import argparse
import math
import os
import shutil
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from models import *

warnings.filterwarnings("ignore")


torch.backends.cudnn.enabled = False 

def configure_optimizers(model, args):
    parameters = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    # Make sure we don't have an intersection of parameters
    params_dict = dict(model.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )

    return optimizer, aux_optimizer


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target, type='train'):
        N, C, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["distortion"] = self.lmbda * out["mse_loss"] * 255**2    

        out["loss"] = out["distortion"] + out["bpp_loss"]


        return out

def combine_gradients(grads_rate, grads_dist, weights):
    """Combine two gradient lists using given weights (Solution 1 or 2)."""
    return [weights[0] * g1 + weights[1] * g2 for g1, g2 in zip(grads_rate, grads_dist)]


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, train_step, tb_writer=None,
                    clip_max_norm=None):
    model.train()
    # device = next(model.parameters()).device

    train_size = 0
    for x in train_dataloader:
        x = x.to('cuda').contiguous()

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out = model(x)

        out_criterion = criterion(out, x, type='train')
        # out_criterion["loss"].backward()

        # --- 3. Get gradients for rate ---
        out_criterion["bpp_loss"].backward(retain_graph=True)
        grads_rate = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) 
                    for p in model.parameters()]
        optimizer.zero_grad()

        # --- 4. Get gradients for distortion ---
        out_criterion["distortion"].backward(retain_graph=True)
        grads_dist = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) 
                    for p in model.parameters()]
        optimizer.zero_grad()

        # --- 5. Compute the balanced weights ---
        # For demonstration: uniform weights; replace with Solution 1 or 2 weights
        eps = 1e-8
        v1 = torch.cat([g.view(-1) / (out_criterion["bpp_loss"].item() + eps) for g in grads_rate])
        v2 = torch.cat([g.view(-1) / (out_criterion["distortion"].item() + eps) for g in grads_dist])
        Q = torch.tensor([
            [torch.dot(v1, v1), torch.dot(v1, v2)],
            [torch.dot(v2, v1), torch.dot(v2, v2)]
        ], device=v1.device)
        one = torch.ones(2, device=v1.device)
        Q_inv = torch.inverse(Q + 1e-8 * torch.eye(2, device=v1.device))
        lam = 1.0 / (one @ Q_inv @ one)
        w = lam * (Q_inv @ one)
        weights = F.softmax(w, dim=0)
        # Example: If implementing Solution 1 or 2, update weights each step

        # --- 6. Combine gradients ---
        combined_grads = combine_gradients(grads_rate, grads_dist, weights)

        # --- 7. Set combined gradients to each parameter ---
        for p, g in zip(model.parameters(), combined_grads):
            p.grad = g.clone()


        if clip_max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        # optimizer.step(out_criterion["bpp_loss"], out_criterion["mse_loss"])
        optimizer.step()

        if torch.cuda.device_count() > 1:
            aux_loss = model.module.aux_loss()
        else:
            aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        train_step += 1
        if tb_writer and train_step % 10 == 1:
            # tb_writer.add_scalar('train origin loss', out_criterion["origin_loss"].item(), train_step)
            tb_writer.add_scalar('train loss', out_criterion["loss"].item(), train_step)
            tb_writer.add_scalar('train mse', out_criterion["mse_loss"].item(), train_step)
            # tb_writer.add_scalar('train wasserstein Distance', out_criterion["wasserstein"].item(), train_step)
            tb_writer.add_scalar('train img bpp', out_criterion["bpp_loss"].item(), train_step)
            if torch.cuda.device_count() > 1:
                tb_writer.add_scalar('train aux', model.module.aux_loss().item(), train_step)
            else:
                tb_writer.add_scalar('train aux', model.aux_loss().item(), train_step)
            

        train_size += x.shape[0]

    print("train sz:{}".format(train_size))
    return train_step


def eval_epoch(model, criterion, eval_dataloader, epoch, tb_writer=None):
    model.eval()
    # device = next(model.parameters()).device

    loss = 0
    img_bpp = 0
    mse_loss = 0
    aux_loss = []

    if tb_writer:
        save_imgs = True
    else:
        save_imgs = False

    eval_size = 0
    with torch.no_grad():
        for x in eval_dataloader:
            x = x.to('cuda').contiguous()

            out = model(x)
            out_criterion = criterion(out, x, type='eval')

            N, _, H, W = x.shape

            loss += out_criterion["loss"] * N
            img_bpp += out_criterion["bpp_loss"] * N
            mse_loss += out_criterion["mse_loss"] * N
            if torch.cuda.device_count() > 1:
                aux_loss.append(model.module.aux_loss())
            else:
                aux_loss.append(model.aux_loss())

            if save_imgs:
                x_rec = (out["x_hat"] * 255).clamp_(0, 255)
                tb_writer.add_image('input/0', x[0, :, :, :].to(torch.uint8), epoch)
                tb_writer.add_image('input/1', x[1, :, :, :].to(torch.uint8), epoch)
                tb_writer.add_image('input/2', x[2, :, :, :].to(torch.uint8), epoch)
                tb_writer.add_image('output/0', x_rec[0, :, :, :].to(torch.uint8), epoch)
                tb_writer.add_image('output/1', x_rec[1, :, :, :].to(torch.uint8), epoch)
                tb_writer.add_image('output/2', x_rec[2, :, :, :].to(torch.uint8), epoch)
                save_imgs = False

            eval_size += x.shape[0]

        loss = (loss / eval_size).item()
        img_bpp = (img_bpp / eval_size).item()
        mse_loss = (mse_loss / eval_size).item()
        aux_loss = (sum(aux_loss) / len(aux_loss)).item()
        psnr = 10. * np.log10(1. ** 2 / mse_loss)
        if tb_writer:
            tb_writer.add_scalar('eval/eval loss', loss, epoch)
            tb_writer.add_scalar('eval/eval img bpp', img_bpp, epoch)
            tb_writer.add_scalar('eval/eval mse', mse_loss, epoch)
            tb_writer.add_scalar('eval/eval psnr', psnr, epoch)
            tb_writer.add_scalar('eval/eval aux', aux_loss, epoch)

        print("eval sz:{}".format(eval_size))

    print("Epoch(Eval):{}, img bpp:{}, mse:{}, psnr:{}".format(epoch, img_bpp, mse_loss, psnr))

    return loss, img_bpp, mse_loss, psnr, aux_loss


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-4]+"_best"+filename[-4:])

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    # parser.add_argument(
    #     "-m",
    #     "--model",
    #     default="stf",
    #     choices=models.keys(),
    #     help="Model architecture (default: %(default)s)",
    # )
    # parser.add_argument(
    #     "-d", "--dataset", type=str, required=True, help="dataset", default='../fiftyone/open-images-v6'
    # )

    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.0018,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Eval batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--save_path", type=str, default="./ckp_ll", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", default=0, type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--log_dir", default="./logs_ll/", type=str, help="Path to save log")
    parser.add_argument("--recon_dir", default="./test", type=str, help="Test reconstruction image path"
    )
    parser.set_defaults(cuda=True)
    args = parser.parse_args(argv)
    return args

class ImageDataset(data.Dataset):

    def __init__(self, path_dir, img_mode = None, transform=None):
        self.path_dir = path_dir
        self.img_mode = img_mode
        self.transform = transform
        self.images = os.listdir(self.path_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        img_path = os.path.join(self.path_dir, image_name)
        img = Image.open(img_path)

        if self.img_mode is not None:
            img = img.convert(self.img_mode)

        if self.transform is not None:
            img = self.transform(img)

        if img.shape[0] < 3:
            img = img.expand(3,-1,-1)
        elif img.shape[0] > 3:
            img = img[:-1,:,:]
        
        return img



def main(argv):
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    eval_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )  

    train_dataset = ImageDataset("YourTrainDataset",transform=train_transforms)
    eval_dataset = ImageDataset("YourTestDataset", transform=eval_transforms)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    eval_dataloader = data.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # net = models[args.model]()
    net = DemoCompressionModel()
    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        net = nn.DataParallel(net)

    net = net.to(device)
    
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.3)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    tb_writer = SummaryWriter(args.log_dir)
    train_step = 0

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            train_step,
            tb_writer,
            args.clip_max_norm,
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'aux_optimizer': aux_optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'step': train_step,
        }, os.path.join(args.save_path, "ckp"+str(int(args.lmbda*10000))+'.tar'))
        loss, img_bpp, mse_loss, psnr, aux_loss = eval_epoch(net, criterion, eval_dataloader, epoch, tb_writer)


        if torch.cuda.device_count() > 1:
            lr_scheduler.module.step()
        else:
            lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                os.path.join(args.save_path, "ckp"+str(args.lmbda*10000)+'.tar'),
            )

        print("---------------")
    tb_writer.close()

if __name__ == "__main__":
    main(sys.argv[1:])