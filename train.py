# coding:utf8
import tqdm
import os
import time
import json
import sys
import pdb
import traceback
from bdb import BdbQuit
import numpy as np
import torch

from dataloader import get_dataloader
from models.decoder import Decoder
from opts import parse_opt
from self_critical.utils import get_ciderd_scorer, get_self_critical_reward, RewardCriterion


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    opt = parse_opt()
    train_mode = opt.train_mode
    idx2word = json.load(open(opt.idx2word, 'r'))
    captions = json.load(open(opt.captions, 'r'))

    # 模型
    decoder = Decoder(idx2word, opt.settings)
    decoder.to(opt.device)
    lr = opt.learning_rate
    optimizer, xe_criterion = decoder.get_optim_and_crit(lr)
    if opt.resume:
        print("====> loading checkpoint '{}'".format(opt.resume))
        chkpoint = torch.load(opt.resume, map_location=lambda s, l: s)
        assert opt.settings == chkpoint['settings'], \
            'opt.settings and resume model settings are different'
        assert idx2word == chkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        decoder.load_state_dict(chkpoint['model'])
        if chkpoint['train_mode'] == train_mode:
            optimizer.load_state_dict(chkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
        print("====> loaded checkpoint '{}', epoch: {}, train_mode: {}"
              .format(opt.resume, chkpoint['epoch'], chkpoint['train_mode']))
    elif train_mode == 'rl':
        raise Exception('"rl" mode need resume model')

    print('====> process image captions begin')
    word2idx = {}
    for i, w in enumerate(idx2word):
        word2idx[w] = i
    captions_id = {}
    for split, caps in captions.items():
        print('convert %s captions to index' % split)
        captions_id[split] = {}
        for fn, seqs in tqdm.tqdm(caps.items(), ncols=100):
            tmp = []
            for seq in seqs:
                tmp.append([decoder.sos_id] +
                           [word2idx.get(w, None) or word2idx['<UNK>'] for w in seq] +
                           [decoder.eos_id])
            captions_id[split][fn] = tmp
    captions = captions_id
    print('====> process image captions end')

    train_data = get_dataloader(opt.fc_feats, opt.att_feats, captions['train'], decoder.pad_id,
                                opt.max_seq_len, opt.batch_size, opt.num_workers)
    val_data = get_dataloader(opt.fc_feats, opt.att_feats, captions['val'], decoder.pad_id,
                              opt.max_seq_len, opt.batch_size, opt.num_workers, shuffle=False)
    test_captions = {}
    for fn in captions['test']:
        test_captions[fn] = [[]]
    test_data = get_dataloader(opt.fc_feats, opt.att_feats, test_captions, decoder.pad_id,
                               opt.max_seq_len, opt.batch_size, opt.num_workers, shuffle=False)

    if train_mode == 'rl':
        rl_criterion = RewardCriterion()
        ciderd_scorer = get_ciderd_scorer(captions, decoder.sos_id, decoder.eos_id)

    def forward(data, training=True, ss_prob=0.0):
        decoder.train(training)
        loss_val = 0.0
        reward_val = 0.0
        for fns, fc_feats, att_feats, (caps_tensor, lengths), ground_truth in tqdm.tqdm(data, ncols=100):
            fc_feats = fc_feats.to(opt.device)
            att_feats = att_feats.to(opt.device)
            caps_tensor = caps_tensor.to(opt.device)

            if training and train_mode == 'rl':
                sample_captions, sample_logprobs, seq_masks = decoder(
                    fc_feats, att_feats, sample_max=0,
                    max_seq_len=opt.max_seq_len, mode=train_mode)
                decoder.eval()
                with torch.no_grad():
                    greedy_captions, _, _ = decoder(
                        fc_feats, att_feats, sample_max=1,
                        max_seq_len=opt.max_seq_len, mode=train_mode)
                decoder.train(training)
                reward = get_self_critical_reward(
                    sample_captions, greedy_captions, fns, ground_truth,
                    decoder.sos_id, decoder.eos_id, ciderd_scorer)
                loss = rl_criterion(sample_logprobs, seq_masks, torch.from_numpy(reward).float().to(opt.device))
                reward_val += float(np.mean(reward[:, 0]))
            else:
                pred = decoder(fc_feats, att_feats, caps_tensor, ss_prob=ss_prob)
                loss = xe_criterion(pred, caps_tensor[:, 1:], lengths)

            loss_val += float(loss)
            if training:
                optimizer.zero_grad()
                loss.backward()
                clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()

        return loss_val / len(data), reward_val / len(data)

    checkpoint_dir = os.path.join(opt.checkpoint, train_mode)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    result_dir = os.path.join(opt.result, train_mode)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    previous_loss = None
    for epoch in range(opt.max_epochs):
        print('--------------------epoch: %d' % epoch)
        ss_prob = 0.0
        if epoch > opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
        train_loss, train_reward = forward(train_data, ss_prob=ss_prob)
        with torch.no_grad():
            val_loss, _ = forward(val_data, training=False)

        if train_mode == 'xe' and previous_loss is not None and val_loss > previous_loss:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = val_loss

        if epoch in [0, 5, 10, 15, 20, 25, 29, 30, 35, 39, 40, 45, 49]:
            # test
            results = []
            for fns, fc_feats, att_feats, _, _ in tqdm.tqdm(test_data, ncols=100):
                fc_feats = fc_feats.to(opt.device)
                att_feats = att_feats.to(opt.device)
                for i, fn in enumerate(fns):
                    fc_feat = fc_feats[i]
                    att_feat = att_feats[i]
                    with torch.no_grad():
                        rest, _ = decoder.sample(fc_feat, att_feat, beam_size=opt.beam_size, max_seq_len=opt.max_seq_len)
                    results.append({'image_id': fn, 'caption': rest[0]})
            json.dump(results, open(os.path.join(result_dir, 'result_%d.json' % epoch), 'w'))

            chkpoint = {
                'epoch': epoch,
                'model': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': opt.settings,
                'idx2word': idx2word,
                'train_mode': train_mode,
            }
            checkpoint_path = os.path.join(checkpoint_dir, 'model_%d_%.4f_%s.pth' % (
                epoch, val_loss, time.strftime('%m%d-%H%M')))
            torch.save(chkpoint, checkpoint_path)

        print('train_loss: %.4f, train_reward: %.4f, val_loss: %.4f' % (train_loss, train_reward, val_loss))


if __name__ == '__main__':
    try:
        train()
    except BdbQuit:
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
