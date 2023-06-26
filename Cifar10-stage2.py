import os
import argparse
import ssl
import sys
ssl._create_default_https_context = ssl._create_unverified_context
from tqdm import tqdm
import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, STL10
from torch.utils.data import DataLoader
from model.resnet import ResNet18
from model.Multi_fc import Multi_head_fc
from tools.normalize import Normalize
from model.npc_model import NonParametricClassifier
from tools.averagetracker import AverageTracker
from tools.tools import check_clustering_metrics
from losses.Loss_IMI import Loss_IMI_cifar
from losses.Loss_ID import Loss_ID
from losses.Loss_FMI import Loss_FMI_cifar
from losses.Loss_CMI import Loss_CMI
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from lib.protocols import test

class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target, index


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpus", type=str, default="4")
    parser.add_argument("-n", "--num_workers", type=int, default=8)
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=2000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument("--pretrained", default="", type=str, help="path to pretrained models")
    parser.add_argument("--checkpoint", default="", type=str, help="path to save models")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    return args

def main():
    args = parse()
    seed = torch.seed()
    print("the seed: ", seed)
    batch_size, epochs = args.batch_size, args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform1 = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


    train_data = CIFAR10Pair(root='data', train=True, transform=train_transform1, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    low_dim = 128
    class_num = 10
    net = ResNet18(low_dim=low_dim)
    fc = Multi_head_fc(class_num, low_dim)
    norm = Normalize(2)
    npc = NonParametricClassifier(input_dim=low_dim,
                                  output_dim=len(train_data),
                                  tau=1.0,
                                  momentum=0.5)
    loss_id_func = Loss_ID(tau2=2.0)
    loss_fmi_func = Loss_FMI_cifar()

    net, norm, fc = net.to(device), norm.to(device), fc.to(device)
    npc, loss_id, loss_fmi, loss_cri = npc.to(device), loss_id_func.to(device), loss_fmi_func.to(device)
    # lr = 0.03
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=4e-5,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=False,
                                dampening=0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            [5, 10, 20, 30, 50, 100],
                                                        gamma=0.5)

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net,
                                    device_ids=range(len(
                                        args.gpus.split(","))))
        torch.backends.cudnn.benchmark = True

    trackers = {n: AverageTracker() for n in ["loss", "loss_id", "loss_fmi", "loss_imi", "loss_cmi"]}


    if os.path.exists(args.pretrained):
        print('Restart from checkpoint {}'.format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        # optimizer.load_state_dict(checkpoint['optimizer'])
        net.load_state_dict(checkpoint['model'])
        # npc.load_state_dict(checkpoint['npc'])
        net.cuda()
        npc.cuda()
        print("successfully load the checkpoint")
        start_epoch = checkpoint['epoch']

    else:
        print('No checkpoint file at {}'.format(args.pretrained))
        start_epoch = 0
        net = net.cuda()

    file_name = "Cifar10-stage2.py"
    results = {'Epochs': [], 'loss_id': [],'loss_fmi':[],'loss_imi':[], 'loss_cri':[], 'k_means_acc': [], 'k_means_nmi': [],
               'k_means_ari': []}
    save_name_pre = '{}_{}_{}'.format(file_name, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')

    tmp = 1
    loss_list = [0, 0, 0, 0, 0]
    for epoch in range(start_epoch+1, epochs + 1):
            net.train()
            fc.train()
            total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)
            for pos_1, pos_2, target, index in train_bar:
                optimizer.zero_grad()
                inputs_1 = pos_1.to(device, dtype=torch.float32, non_blocking=True)
                inputs_2 = pos_2.to(device, dtype=torch.float32, non_blocking=True)
                indexes = index.to(device, non_blocking=True)
                features_1 = norm(net(inputs_1))
                features_2 = norm(net(inputs_2))
                outputs = npc(features_1, indexes)
                features = torch.cat((features_1, features_2), 0)
                tmp_list = fc(features)

                loss_imi = Loss_IMI_cifar(features_1, features_2)

                ## the main code of our method will be attached here after acceptance


                tot_loss = loss_id + 0.01 * loss_fmi + 0.0001 * loss_imi + 2 * loss_cmi
                tot_loss.backward()

                optimizer.step()
                # track loss
                trackers["loss"].add(tot_loss.item())
                trackers["loss_id"].add(loss_id.item())
                trackers["loss_imi"].add(loss_imi.item())
                trackers["loss_fmi"].add(loss_fmi.item())
                trackers["loss_cmi"].add(loss_cmi.item())
            lr_scheduler.step()

            if (epoch == 0) or (((epoch + 1) % 1) == 0):
                acc, nmi, ari = check_clustering_metrics(npc, train_loader)
                print("lr = ", lr_scheduler.get_last_lr())
                print("Epoch:{} Loss_id:{} Loss_mr:{} Loss_cl:{} loss_cmi:{} Kmeans ACC, NMI, ARI = {}, {}, {}"
                      "".format(epoch+1, loss_id, loss_fmi, loss_imi, loss_cmi, acc, nmi, ari))
                results['Epochs'].append(tmp * 1)
                results['loss_id'].append(loss_id.item())
                results['loss_fmi'].append(loss_fmi.item())
                results['loss_imi'].append(loss_imi.item())
                results['loss_cmi'].append(loss_cmi.item())
                results['k_means_acc'].append(acc)
                results['k_means_nmi'].append(nmi)
                results['k_means_ari'].append(ari)
                # Checkpoint
                print('Checkpoint ...')
                torch.save({'optimizer': optimizer.state_dict(), 'model': net.state_dict(),
                            'epoch': epoch + 1, 'npc': npc.state_dict()}, args.checkpoint)

                tmp = tmp + 1
                # save statistics   df = pd.DataFrame.from_dict(d, orient='index')
                data_frame = pd.DataFrame.from_dict(data=results, orient='index')
                data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre))



if __name__ == "__main__":
    main()
