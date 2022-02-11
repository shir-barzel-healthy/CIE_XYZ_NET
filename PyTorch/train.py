"""
 Copyright 2020 Mahmoud Afifi.
 Released under the MIT License.
 If you use this code, please cite the following paper:
 Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith
 Punnappurath, and Michael S Brown.
 CIE XYZ Net: Unprocessing Images for Low-Level Computer Vision Tasks.
 arXiv preprint, 2020.
"""

__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import argparse
import logging
import os
import sys
from matplotlib.pyplot import margins
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from src import sRGB2XYZ
import src.utils as utls
from src import utils


from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    use_tb = True
except ImportError:
    use_tb = False

from src.dataset import BasicDataset
from torch.utils.data import DataLoader

def train_net(net, device, dir_img, dir_gt, val_dir, val_dir_gt, epochs=300,
              batch_size=4, lr=0.0001, lrdf=0.5, lrdp=75, l2reg=0.001,
              chkpointperiod=1, patchsz=256, validationFrequency=10,
              save_cp=True, state='orig', start_epoch=0, optimizer=None, scheduler=None, pre_state=None, desc=None):

    if state == 'orig':
        batch_size = batch_size * 2

    dir_checkpoint = 'checkpoints/'

    train = BasicDataset(dir_img, dir_gt, patch_size=patchsz, state=state)
    val = BasicDataset(val_dir, val_dir_gt, patch_size=patchsz, state=state)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True, drop_last=True)
    if use_tb:
        if pre_state:
            writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_state_{state}_{pre_state}')
        else:
            writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_state_{state}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs} epochs
        Batch size:      {batch_size}
        Patch size:      {patchsz} x {patchsz}
        Learning rate:   {lr}
        Training size:   {len(train)}
        Validation size: {len(val)}
        Validation Frq.: {validationFrequency}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        TensorBoard:     {use_tb}
    ''')
    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=l2reg)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, lrdp, gamma=lrdf,
                                          last_epoch=-1)

    now = datetime.now() # current date and time
    date_time = now.strftime("%m_%d_%H_%M")

    print(f"\nDate: {date_time}\n")

    # des = input("Enter description: ")
    f = open("runs.txt", "a")
    if desc is None:
        print("no desc")
        exit(1)
    f.write(f"{date_time}: {desc}\n")
    f.close()


    for epoch in range(start_epoch, epochs):
        net.train()

        epoch_loss_imgs = 0
        epoch_loss_m = 0
        with tqdm(total=len(train), desc=f'Epoch {epoch + 1}/{epochs}',
                  unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                xyz_gt = batch['gt_xyz']
                if state == 'orig':
                    assert_val_imgs = imgs.shape[1] == 3
                    assert_val_xyz = xyz_gt.shape[1] == 3
                elif state == 'self-sup':
                    assert_val_imgs = imgs.shape[2] == 3
                    assert_val_xyz = xyz_gt.shape[2] == 3
                assert assert_val_imgs, (
                    f'Network has been defined with 3 input channels, '
                    f'but loaded training images have {imgs.shape}.'
                    f' Please check that the images are loaded correctly.')

                assert assert_val_xyz, (
                    f'Network has been defined with 3 input channels, '
                    f'but loaded XYZ images have {xyz_gt.shape}. '
                    f'Please check that the images are loaded correctly.')

                imgs = imgs.to(device=device, dtype=torch.float32)
                xyz_gt = xyz_gt.to(device=device, dtype=torch.float32)
                if state == 'orig':
                    rec_imgs, rendered_imgs = net(imgs)
                    loss_imgs = utls.compute_loss(imgs, xyz_gt, rec_imgs, rendered_imgs)
                elif state == 'self-sup':
                    rec_1, rend_1, m_inv_1, m_fwd_1, rec_2, rend_2, m_inv_2, m_fwd_2 = net(imgs)
                    loss_imgs, loss_m = utls.compute_loss_self_sup(imgs, xyz_gt, rec_1, rend_1, m_inv_1,
                                            m_fwd_1, rec_2, rend_2, m_inv_2, m_fwd_2)

                epoch_loss_imgs += loss_imgs.item()
                if state == 'self-sup':
                    epoch_loss_m += loss_m.item()

                if use_tb:
                    writer.add_scalar('Loss_imgs/train', loss_imgs.item(), epoch)

                pbar.set_postfix(**{'loss_imgs (batch)': loss_imgs.item()})

                if state == 'self-sup':
                    optimizer.zero_grad()
                    m_fac = 100
                    loss_m = loss_m * m_fac
                    if pre_state == 'only-self-sup':
                        loss_m.backward()
                    else:
                        loss_m.backward(retain_graph=True)
                        loss_imgs.backward()
                    optimizer.step()
                elif state == 'orig':
                    optimizer.zero_grad()
                    loss_imgs.backward()
                    optimizer.step()

                pbar.update(np.ceil(imgs.shape[0]))
                global_step += 1

        writer.add_scalar('Loss_m/train', epoch_loss_m / len(train), epoch)
        writer.add_scalar('Loss_imgs/train', epoch_loss_imgs / len(train), epoch)
        if (epoch + 1) % validationFrequency == 0:
            val_score_imgs, val_score_m, mean_psnr_srgb, mean_psnr_xyz, matrices = vald_net(net, val_loader, device, state)
            logging.info('Validation loss_imgs: {}'.format(val_score_imgs))
            logging.info('Validation loss_m: {}'.format(val_score_m))
            if use_tb:
                writer.add_scalar('learning_rate',
                                  optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Loss_imgs/val', val_score_imgs, epoch)
                writer.add_scalar('Loss_m/val', val_score_m, epoch)
                writer.add_scalar('Mean_Val_PSNR_XYZ', mean_psnr_xyz, epoch)
                writer.add_scalar('Mean_Val_PSNR_SRGB', mean_psnr_srgb, epoch)
                if state == 'orig':
                    writer.add_images('images', imgs, epoch)
                    writer.add_images('rendered-imgs', rendered_imgs, epoch)
                    writer.add_images('rec-xyz', rec_imgs, epoch)
                    writer.add_images('gt-xyz', xyz_gt, epoch)
                elif state == 'self-sup':
                    writer.add_images('image-1', torch.squeeze(imgs[:, 0, :, :, :]), epoch)
                    writer.add_images('gt-xyz-1', torch.squeeze(xyz_gt[:, 0, :, :, :]), epoch)
                    writer.add_images('rec-xyz_1', rec_1, epoch)
                    writer.add_images('rendered-imgs_1', rend_1, epoch)
                    writer.add_images('image-2', torch.squeeze(imgs[:, 1, :, :, :]), epoch)
                    writer.add_images('gt-xyz-2', torch.squeeze(xyz_gt[:, 1, :, :, :]), epoch)
                    writer.add_images('rendered-imgs_2', rend_2, epoch)
                    writer.add_images('rec-xyz_2', rec_2, epoch)

        scheduler.step()

        if save_cp and (epoch + 1) % chkpointperiod == 0:
            if not os.path.exists(dir_checkpoint):
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')

            torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_imgs, 'matrices': matrices },
                     dir_checkpoint + f'f_ciexyznet_{state}_{date_time}_{epoch + 1}.pth')
            # torch.save(net.state_dict(), dir_checkpoint +
            #            f'ciexyznet_{state}_{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved!')

    if not os.path.exists('models'):
        os.mkdir('models')
        logging.info('Created trained models directory')
    torch.save(net.state_dict(), 'models/' + 'model_sRGB-XYZ-sRGB.pth')
    logging.info('Saved trained model!')
    if use_tb:
        writer.close()
    logging.info('End of training')


def vald_net(net, loader, device, state):
    """Evaluation using MAE"""
    net.eval()
    n_val = len(loader) + 1
    mae_imgs = 0
    mae_m = 0

    mean_psnr_xyz_list = []
    mean_psnr_srgb_list = []

    matrices = []
    with tqdm(total=n_val, desc='Validation round', unit='batch',
              leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            xyz_gt = batch['gt_xyz']
            if state == 'orig':
                assert_val_imgs = imgs.shape[1] == 3
                assert_val_xyz = xyz_gt.shape[1] == 3
            elif state == 'self-sup':
                assert_val_imgs = imgs.shape[2] == 3
                assert_val_xyz = xyz_gt.shape[2] == 3
            assert assert_val_imgs, (
                f'Network has been defined with 3 input channels, '
                f'but loaded training images have {imgs.shape}.'
                f' Please check that the images are loaded correctly.')

            assert assert_val_xyz, (
                f'Network has been defined with 3 input channels, '
                f'but loaded XYZ images have {xyz_gt.shape}. '
                f'Please check that the images are loaded correctly.')

            imgs = imgs.to(device=device, dtype=torch.float32)
            xyz_gt = xyz_gt.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                if state == 'orig':
                    rec_imgs, rendered_imgs = net(imgs)

                    batch_psnr_xyz = []
                    batch_psnr_srgb = []
                    for i in range(rec_imgs.shape[0]):
                        psnr_xyz = utils.PSNR(255 * xyz_gt[i], 255 * rec_imgs[i])
                        psnr_srgb = utils.PSNR(255 * imgs[i], 255 * rendered_imgs[i])
                        batch_psnr_xyz.append(psnr_xyz)
                        batch_psnr_srgb.append(psnr_srgb)
                    mean_batch_psnr_xyz = np.mean(np.array(batch_psnr_xyz))
                    mean_psnr_xyz_list.append(mean_batch_psnr_xyz)
                    mean_batch_psnr_srgb = np.mean(np.array(batch_psnr_srgb))
                    mean_psnr_srgb_list.append(mean_batch_psnr_srgb)

                    loss_imgs = utls.compute_loss(imgs, xyz_gt, rec_imgs, rendered_imgs)
                elif state == 'self-sup':
                    rec_1, rend_1, m_inv_1, m_fwd_1, rec_2, rend_2, m_inv_2, m_fwd_2 = net(imgs)

                    matrices.append([(m_inv_1,m_inv_2),(m_fwd_1, m_fwd_2)])
                    batch_psnr_xyz = []
                    batch_psnr_srgb = []
                    xyz_gt_1 = torch.squeeze(xyz_gt[:, 0, :, :, :])
                    xyz_gt_2 = torch.squeeze(xyz_gt[:, 1, :, :, :])
                    imgs_gt_1 = torch.squeeze(imgs[:, 0, :, :, :])
                    imgs_gt_2 = torch.squeeze(imgs[:, 1, :, :, :])

                    for i in range(rec_1.shape[0]):
                        psnr_xyz_1 = utils.PSNR(255 * xyz_gt_1[i], 255 * rec_1[i])
                        psnr_xyz_2 = utils.PSNR(255 * xyz_gt_2[i], 255 * rec_2[i])
                        psnr_srgb_1 = utils.PSNR(255 * imgs_gt_1[i], 255 * rend_1[i])
                        psnr_srgb_2 = utils.PSNR(255 * imgs_gt_2[i], 255 * rend_2[i])
                        batch_psnr_xyz.append(psnr_xyz_1)
                        batch_psnr_xyz.append(psnr_xyz_2)
                        batch_psnr_srgb.append(psnr_srgb_1)
                        batch_psnr_srgb.append(psnr_srgb_2)
                    mean_batch_psnr_xyz = np.mean(np.array(batch_psnr_xyz))
                    mean_psnr_xyz_list.append(mean_batch_psnr_xyz)
                    mean_batch_psnr_srgb = np.mean(np.array(batch_psnr_srgb))
                    mean_psnr_srgb_list.append(mean_batch_psnr_srgb)

                    loss_imgs, loss_m = utls.compute_loss_self_sup(imgs, xyz_gt, rec_1, rend_1, m_inv_1,
                                        m_fwd_1, rec_2, rend_2, m_inv_2, m_fwd_2)
                                        
                mae_imgs = mae_imgs + loss_imgs
                if state == 'self-sup':
                    mae_m = mae_m + loss_m
                else:
                    mae_m = 0

            pbar.update(np.ceil(imgs.shape[0]))

    net.train()
    return mae_imgs / n_val, mae_m / n_val, np.mean(np.array(mean_psnr_srgb_list)), np.mean(np.array(mean_psnr_xyz_list)), matrices

def get_args():
    parser = argparse.ArgumentParser(description='Train CIE XYZ Net.')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=300,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?',
                        default=4, help='Batch size', dest='batchsize')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float,
                        nargs='?', default=0.0001, help='Learning rate',
                        dest='lr')
    parser.add_argument('-l2r', '--l2reg', metavar='L2Reg', type=float,
                        nargs='?', default=0.001, help='L2 Regularization '
                                                       'factor', dest='l2r')
    parser.add_argument('-l', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-vf', '--validation-frequency', dest='val_frq',
                        type=int, default=5, help='Validation frequency.')
    parser.add_argument('-s', '--patch-size', dest='patchsz', type=int,
                        default=256, help='Size of training patch')
    parser.add_argument('-c', '--checkpoint-period', dest='chkpointperiod',
                        type=int, default=10,
                        help='Number of epochs to save a checkpoint')
    parser.add_argument('-ldf', '--learning-rate-drop-factor', dest='lrdf',
                        type=float, default=0.5,
                        help='Learning rate drop factor')
    parser.add_argument('-ldp', '--learning-rate-drop-period', dest='lrdp',
                        type=int, default=75, help='Learning rate drop period')
    parser.add_argument('-ntrd', '--training_dir_in', dest='in_trdir',
                        default='../sRGB2XYZ_training/sRGB_training/',
                        help='Input training image directory')
    parser.add_argument('-gtrd', '--training_dir_gt', dest='gt_trdir',
                        default='../sRGB2XYZ_training/XYZ_training/',
                        help='Ground truth training image directory')
    parser.add_argument('-nvld', '--validation_dir_in', dest='in_vldir',
                        default='../sRGB2XYZ_validation/sRGB_validation/',
                        help='Input validation image directory')
    parser.add_argument('-gvld', '--validation_dir_gt', dest='gt_vldir',
                        default='../sRGB2XYZ_validation/XYZ_validation/',
                        help='Ground truth validation image directory')
    parser.add_argument('-state', '--net_state', dest='state',
                        default='orig',
                        help='self supervised or original state')
    parser.add_argument('-starte', '--start_epoch', dest='start_epoch',
                        type=int, default=0,
                        help='starting epoch')
    parser.add_argument('-prestate', '--pre_state', dest='pre_state',
                        default=None,
                        help='pre_state')
    parser.add_argument('-des', '--description', dest='desc',
                        default=None,
                        help='desc')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Training of CIE XYZ Net')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # net = nn.DataParallel(sRGB2XYZ.CIEXYZNet(device=device, state=args.state))
    net = sRGB2XYZ.CIEXYZNet(device=device, state=args.state)
    # if args.load:
    #     optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),
    #                        eps=1e-08, weight_decay=args.l2r)
    #     checkpoint = torch.load(args.load)
    #     net.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch']
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, args.lrdp, gamma=args.lrdf,
    #                                 last_epoch=-start_epoch)
    #     # net.load_state_dict(
    #     #     torch.load(args.load, map_location=device)
    #     # )
    #     logging.info(f'Model loaded from {args.load}')
    # else:
    #     optimizer = None
    #     scheduler = None
    
    optimizer = None
    scheduler = None
    if args.load:
        logging.info(f'Model loaded from {args.load}')
        checkpoint = torch.load(args.load)
        net.load_state_dict(checkpoint['model_state_dict'])

    net.to(device=device)

    try:
        print(f"\n\n\nUsing state: {args.state}\n\n\n")
        train_net(net=net, device=device, dir_img=args.in_trdir,
                  dir_gt=args.gt_trdir, val_dir=args.in_vldir,
                  val_dir_gt=args.gt_vldir, epochs=args.epochs,
                  batch_size=args.batchsize, lr=args.lr, lrdf=args.lrdf,
                  lrdp=args.lrdp, chkpointperiod=args.chkpointperiod,
                  validationFrequency=args.val_frq, patchsz=args.patchsz, state=args.state, start_epoch=args.start_epoch,
                  optimizer=optimizer, scheduler=scheduler, pre_state=args.pre_state, desc=args.desc)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'intrrupted_check_point.pth')
        logging.info('Saved interrupt checkpoint backup')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
