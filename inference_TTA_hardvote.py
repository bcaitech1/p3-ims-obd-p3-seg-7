import torch
import numpy as np
import argparse
import json
import os
from importlib import import_module
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import FCN8s
from utils import get_category_names
from load_data import CustomDataLoader


def hard_voting(target):
    print(target.shape)
    N, S = target.shape[1], target.shape[2]
    result = np.zeros_like(target[0])
    target = np.transpose(target, [2, 1, 0])
    for i in range(N):
        for j in range(S):
            result[i][j] = np.bincount(target[j][i]).argmax()

    print(result.shape)

    return result




def inference(model_name, model_num, batch_size, num_workers, type):
    # define model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_module = getattr(import_module('model'), args.model)
    model = model_module(num_classes=12).to(device)

    # load test_loader
    dataset_path = '../data'
    test_path = dataset_path + '/test.json'

    if model_num == '1':
        model_num = ''
    else:
        model_num = f'_{model_num}'

    # best model 저장된 경로
    model_path = os.path.join('./results', f"{model_name}{model_num}", f"{model_name}_best_{type}.pt")
    anns_file_path = dataset_path + '/' + 'train.json'

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    category_names = get_category_names(dataset)

    def collate_fn(batch):
        return tuple(zip(*batch))

    # transform_module = getattr(import_module("augmentation"), args.augmentation)
    # test_transform = transform_module()

    test_dataset = CustomDataLoader(dataset_path=dataset_path, data_dir=test_path, category_names=category_names, mode='test')

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            collate_fn=collate_fn
                                            )

    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    aug_transform = [A.Compose([]), A.Compose([A.HorizontalFlip(1.0)]), A.Compose([A.VerticalFlip(1.0)]), A.Compose([A.HorizontalFlip(1.0), A.VerticalFlip(1.0)])]
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = [np.empty((0, size*size), dtype=np.long)] * len(aug_transform)
    
    with torch.no_grad():
        for i in range(len(aug_transform)):
            for step, (raw_imgs, image_infos) in enumerate(test_loader):
                imgs = []
                for img in raw_imgs:
                    img = aug_transform[i](image=img)['image']
                    imgs.append(img)
                imgs = np.stack(imgs)
                imgs = imgs.transpose([0,3,1,2])

                # inference (512 x 512)
                outs = model(torch.from_numpy(imgs).to(device))
                oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()

                
                # resize (256 x 256)
                temp_mask = []
                for img, mask in zip(np.stack(imgs), oms):
                    mask = transform(image=img, mask=mask)['mask']
                    mask = aug_transform[i](image=img, mask=mask)['mask']
                    temp_mask.append(mask)

                oms = np.array(temp_mask, dtype=np.float16)
                
                oms = oms.reshape([oms.shape[0], size*size]).astype(int)
                preds_array[i] = np.vstack((preds_array[i], oms))
                
                if i == 0:
                    file_name_list.append([i['file_name'] for i in image_infos])

            print(f"augmentation {i} ended.")
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]

    print("so far so good")
    preds_array = hard_voting(np.array(preds_array))
    print("perfect ending!")
    # sample_submisson.csv 열기
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(f"./submission/{model_name}{model_num}_{type}_TTA_hardvote.csv", index=False)


def main(args):
    inference(args.model, args.model_num, args.batch_size, args.num_workers, args.type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
  
    # model dir
    # FCN8s / DeconvNet / SegNet / Deeplab_V3_Resnet101
    parser.add_argument('--model', type=str, default="Deeplab_V3_Resnet101")
    parser.add_argument('--model_num', type=str, default="1")
    parser.add_argument('--augmentation', type=str, default='BasicAugmentation', help='augmentation method for training')
    parser.add_argument('--type', type=str, default='loss')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (deafult: 8)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader (default: 4)')
    args = parser.parse_args()
    main(args)