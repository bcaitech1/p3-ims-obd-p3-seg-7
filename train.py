import os
import json
import torch
import argparse
import numpy as np
from importlib import import_module
import sys

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import *
from load_data import CustomDataLoader
# from model import FCN8s


def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(device), masks.to(device)            

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

            mIoU = label_accuracy_score(masks.detach().cpu().numpy(), outputs, n_class=12)[2]
            mIoU_list.append(mIoU)
            
        avrg_loss = total_loss / cnt
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss, np.mean(mIoU_list)))

    return avrg_loss


def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, device, model_name):
    def save_model(model, saved_dir, file_name='fcn8s_best_model(pretrained).pt'):
        check_point = {'net': model.state_dict()}
        output_path = os.path.join(saved_dir, file_name)
        torch.save(model.state_dict(), output_path)
    # start training
    print('Start training..')
    best_loss = 9999999
    for epoch in range(num_epochs):
        model.train()
        for step, (images, masks, _) in enumerate(data_loader):
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
                  
            # inference
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, num_epochs, step+1, len(data_loader), loss.item()))
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
            if avrg_loss < best_loss:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_loss = avrg_loss

                # save best model
                check_point = {'net': model.state_dict()}
                output_path = os.path.join(saved_dir, model_name + '.pt')
                torch.save(model.state_dict(), output_path)


def main(args):
    # fix seed
    seed_everything(seed=1024)

    # defind save directory
    save_dir = increment_path(os.path.join(args.model_dir, args.model))
    os.makedirs(save_dir, exist_ok=True)

    # save args on .json file
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # open sys logging
    consoleLog = Logger(save_dir=save_dir)
    sys.stdout = consoleLog

    # set dataset
    dataset_path = '../data'
    anns_file_path = dataset_path + '/' + 'train.json'

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    category_names = get_category_names(dataset)

    # train.json / validation.json / test.json
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'
    test_path = dataset_path + '/test.json'

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_transform = A.Compose([
                                ToTensorV2()
                                ])

    val_transform = A.Compose([
                            ToTensorV2()
                            ])

    test_transform = A.Compose([
                            ToTensorV2()
                            ])

    # validation set을 직접 나누고 싶은 경우
    # random_split 사용하여 data set을 8:2 로 분할
    # train_size = int(0.8*len(dataset))
    # val_size = int(len(dataset)-train_size)
    # dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # create own Dataset 2
    # train dataset
    train_dataset = CustomDataLoader(dataset_path=dataset_path, data_dir=train_path, category_names=category_names, mode='train', transform=train_transform)

    # validation dataset
    val_dataset = CustomDataLoader(dataset_path=dataset_path, data_dir=val_path, category_names=category_names, mode='val', transform=val_transform)

    # test dataset
    test_dataset = CustomDataLoader(dataset_path=dataset_path, data_dir=test_path, category_names=category_names, mode='test', transform=test_transform)


    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            collate_fn=collate_fn)

    # define model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_module = getattr(import_module('model'), args.model)
    model = model_module(num_classes=12).to(device)

    # define optimizer  & criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # train model
    val_every = 1
    train(args.epochs, model, train_loader, val_loader, criterion, optimizer, save_dir, val_every, device, args.model)

    # close sys logging
    del consoleLog


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=1024, help='random seed (default: 1024)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs for train (deafult: 20)')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (deafult: 16)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader (default: 4)')
    parser.add_argument('--smoothing', type=float, default=0.2, help='label smoothing facotr for label smoothing loss (default: 0.2)')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay (default: 1e-6)')

    parser.add_argument('--model_dir', type=str, default='./results', help='directory where model would be saved (default: ./results)')

    # FCN8s / DeconvNet / SegNet / Deeplab_V3_Resnet101
    parser.add_argument('--model', type=str, default='SegNet', help='backbone bert model for training (default: FCN8s)')
    parser.add_argument('--valid', type=int, default=1, help='whether split training set (default: 1)')

    args = parser.parse_args()

    main(args)