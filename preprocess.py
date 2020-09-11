import argparse
import json
import tqdm
import h5py
import skimage.io
import os
from collections import Counter
import torch

from models.encoder import Encoder


def extract_imgs_feat():
    encoder = Encoder(opt.resnet101_file)
    encoder.to(opt.device)
    encoder.eval()

    imgs = os.listdir(opt.imgs_dir)
    imgs.sort()

    if not os.path.exists(opt.out_feats_dir):
        os.makedirs(opt.out_feats_dir)
    with h5py.File(os.path.join(opt.out_feats_dir, '%s_fc.h5' % opt.dataset_name)) as file_fc, \
            h5py.File(os.path.join(opt.out_feats_dir, '%s_att.h5' % opt.dataset_name)) as file_att:
        try:
            for img_nm in tqdm.tqdm(imgs, ncols=100):
                img = skimage.io.imread(os.path.join(opt.imgs_dir, img_nm))
                with torch.no_grad():
                    img = encoder.preprocess(img)
                    img = img.to(opt.device)
                    img_fc, img_att = encoder(img)
                file_fc.create_dataset(img_nm, data=img_fc.cpu().float().numpy())
                file_att.create_dataset(img_nm, data=img_att.cpu().float().numpy())
        except BaseException as e:
            file_fc.close()
            file_att.close()
            print('--------------------------------------------------------------------')
            raise e


def process_coco_captions():
    images = json.load(open(opt.dataset_coco, 'r'))['images']
    captions = {'train': {}, 'val': {}, 'test': {}}
    annotation = {}
    idx2word = Counter()
    for img in tqdm.tqdm(images, ncols=100):
        split = 'train'
        if img['split'] == 'val':
            split = 'val'
        elif img['split'] == 'test':
            split = 'test'
        sents = []
        rows = []
        for sent in img['sentences']:
            sents.append(sent['tokens'])
            idx2word.update(sent['tokens'])
            rows.append(sent['raw'].lower().strip())
        captions[split][img['filename']] = sents
        if split == 'test':
            annotation[img['filename']] = rows
    json.dump(captions, open(opt.out_captions, 'w'))
    json.dump(annotation, open(opt.out_annotation, 'w'))

    idx2word = idx2word.most_common()
    idx2word = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + [w[0] for w in idx2word if w[1] > 4]
    json.dump(idx2word, open(opt.out_idx2word, 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='coco')
    parser.add_argument('--imgs_dir', type=str, default='./data/images/')
    parser.add_argument('--resnet101_file', type=str,
                        default='./data/pre_models/resnet101.pth')
    parser.add_argument('--out_feats_dir', type=str, default='./data/features/')

    parser.add_argument('--dataset_coco', type=str, default='../../dataset/caption/coco/dataset_coco.json')
    parser.add_argument('--out_captions', type=str, default='./data/captions/captions.json')
    parser.add_argument('--out_idx2word', type=str, default='./data/captions/idx2word.json')
    parser.add_argument('--out_annotation', type=str, default='./data/captions/annotation.json')

    opt = parser.parse_args()

    extract_imgs_feat()
    process_coco_captions()
