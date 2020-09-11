import argparse
import torch


def parse_opt():
    parser = argparse.ArgumentParser()

    # train settings
    parser.add_argument('--train_mode', type=str, default='xe', choices=['xe', 'rl'])
    parser.add_argument('--learning_rate', type=float, default=4e-4)  # 4e-4 for xe, 4e-5 for rl
    parser.add_argument('--resume', type=str, default='')  # required for rl
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--scheduled_sampling_start', type=int, default=0)
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5)
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05)
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25)

    parser.add_argument('--idx2word', type=str, default='./data/captions/idx2word.json')
    parser.add_argument('--captions', type=str, default='./data/captions/captions.json')
    parser.add_argument('--fc_feats', type=str, default='./data/features/coco_fc.h5')
    parser.add_argument('--att_feats', type=str, default='./data/features/coco_att.h5')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--result', type=str, default='./result/')
    parser.add_argument('--max_seq_len', type=int, default=16)
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--beam_size', type=int, default=3)

    # eval settings
    parser.add_argument('-e', '--eval_model', type=str, default='')
    parser.add_argument('-r', '--result_file', type=str, default='')

    # test setting
    parser.add_argument('-t', '--test_model', type=str, default='')
    parser.add_argument('-i', '--image_file', type=str, default='')
    # encoder settings
    parser.add_argument('--resnet101_file', type=str, default='./data/pre_models/resnet101.pth',
                        help='Pre-trained resnet101 network for extracting image features')

    args = parser.parse_args()

    # decoder settings
    settings = dict()
    settings['word_emb_dim'] = 512
    settings['fc_feat_dim'] = 2048
    settings['att_feat_dim'] = 2048
    settings['feat_emb_dim'] = 512
    settings['dropout_p'] = 0.5
    settings['rnn_hid_dim'] = 512
    settings['att_hid_dim'] = 512

    args.settings = settings
    args.use_gpu = torch.cuda.is_available()
    args.device = torch.device('cuda:0') if args.use_gpu else torch.device('cpu')
    return args
