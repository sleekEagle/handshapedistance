# Copyright (c) 2020 Graz University of Technology All rights reserved.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import sys
sys.path.append(os.getcwd()) 
from hand_keypoint.common.base import Tester


from hand_keypoint.common.utils.preprocessing import PeakDetector, load_skeleton
from hand_keypoint.common.utils.vis import *
import configargparse
import torch
import torchvision.transforms as transforms


jointsMapManoToDefault = [
                         16, 15, 14, 13,
                         17, 3, 2, 1,
                         18, 6, 5, 4,
                         19, 12, 11, 10,
                         20, 9, 8, 7,
                         0]

VIS_ATTN = False

def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--annot_subset', type=str, dest='annot_subset', default='all',
                        help='all/human_annot/machine_annot')
    parser.add_argument('--test_set', type=str, dest='test_set', default='test', help='Split type (test/train/val)')
    parser.add_argument('--ckpt_path', type=str, default='D:\\data\\trained_models\\keypt_transformer\\interhand_2.5D.tar',dest='ckpt_path', help='Full path to the checkpoint file')
    parser.add_argument('--use_big_decoder', action='store_true', help='Use Big Decoder for U-Net')
    parser.add_argument('--dec_layers', type=int, default=1, help='Number of Cross-attention layers')
    # args = parser.parse_args(['--use_big_decoder'])
    args = parser.parse_args()
    args.capture, args.camera, args.seq_name = None, None, None

    cfg.use_big_decoder = args.use_big_decoder
    cfg.dec_layers = args.dec_layers
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))


    return args

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


class Predict:
    def __init__(self) -> None:
        args = parse_args()
        cfg.set_args(args.gpu_ids, '')
        print(args)

        self.tester = Tester(args.ckpt_path)
        # self.tester._make_batch_generator(args.test_set, args.annot_subset, args.capture, args.camera, args.seq_name)
        self.tester._make_model()
        print('created the model')

        ih26m_joint_regressor = np.load(cfg.joint_regr_np_path)
        ih26m_joint_regressor = torch.FloatTensor(ih26m_joint_regressor).unsqueeze(0)  # 1 x 21 x 778

        self.skeleton = load_skeleton(cfg.skeleton_file, 42)
        self.peak_detector = PeakDetector()
        self.transform = transforms.Compose([transforms.PILToTensor()])

    #hand: a list of eithrt ['left'] or ['write'] or both ['left','rigth']
    #hand_im: the original PIL image of the hand/s
    def draw_skeleton(self,hand_im,joints_2d_right,joints_2d_left,hand_list):
        assert 'right' in hand_list or 'left' in hand_list , "hand-list must include left or right or both" 
        img = (np.asarray(hand_im)).astype(np.uint8)
        img = cv2.resize(img, (128, 128))
        img_2d = img.copy()
        if 'left' in hand_list:
            img_2d = vis_keypoints_new(img_2d, joints_2d_left.cpu().numpy(), np.ones((21)), self.skeleton, line_width=1, circle_rad=1.5, hand_type='left')
        if 'right' in hand_list:
            img_2d = vis_keypoints_new(img_2d, joints_2d_right.cpu().numpy(), np.ones((21)), self.skeleton, line_width=1, circle_rad=1.5, hand_type='right')
        return img_2d

    #inputs : dictionary with keys: img , mask
    #imputs['img'] : bs,3,256,256 images are scaled from 0 to 1
    #inputs['mask'] : bs,1,256,256
    def make_prediction(self,im_list):
        #create a batch of img tensors from the list of PIL images given
        im_t=torch.empty(0,3,256,256)
        for im in im_list:
            img_tensor = self.transform(im)/255.0
            img_tensor=torch.unsqueeze(img_tensor,dim=0)
            im_t=torch.concat((im_t,img_tensor),dim=0)
        # img_tensor=self.transform(im)/255.0
        # img_tensor=torch.unsqueeze(img_tensor,dim=0)
        mask=torch.zeros(len(im_list),1,256,256)
        input={'img':im_t,'mask':mask}
        with torch.no_grad():
            model_out = self.tester.model(input, 'test', epoch_cnt=1e8)
            out = {k[:-4]: model_out[k] for k in model_out.keys() if '_out' in k}
            heatmap_np = out['joint_heatmap'].cpu().numpy()
            joints_2d_right = out['joint_2p5d'][:, :21]
            joints_2d_left = out['joint_2p5d'][:, 21:]
        return heatmap_np,joints_2d_right,joints_2d_left