# coding:utf8
import torch
import skimage.io

from opts import parse_opt
from models.decoder import Decoder
from models.encoder import Encoder

opt = parse_opt()
assert opt.test_model, 'please input test_model'
assert opt.image_file, 'please input image_file'

encoder = Encoder(opt.resnet101_file)
encoder.to(opt.device)
encoder.eval()

img = skimage.io.imread(opt.image_file)
with torch.no_grad():
    img = encoder.preprocess(img)
    img = img.to(opt.device)
    fc_feat, att_feat = encoder(img)

print("====> loading checkpoint '{}'".format(opt.test_model))
chkpoint = torch.load(opt.test_model, map_location=lambda s, l: s)
decoder = Decoder(chkpoint['idx2word'], chkpoint['settings'])
decoder.load_state_dict(chkpoint['model'])
print("====> loaded checkpoint '{}', epoch: {}, train_mode: {}".
      format(opt.test_model, chkpoint['epoch'], chkpoint['train_mode']))
decoder.to(opt.device)
decoder.eval()

rest, _ = decoder.sample(fc_feat, att_feat, beam_size=opt.beam_size,
                         max_seq_len=opt.max_seq_len)
print('generate captions:\n' + '\n'.join(rest))
