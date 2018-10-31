from config.config import cfg
from models.yolo.darknet.darknet import *
from utils import cal_IOU

class YoloCore:

    def __init__(self, gpu='0'):

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        self.thresh = cfg.darknet.thresh
        self.hier_thresh = cfg.darknet.hier_thresh
        self.nms = cfg.darknet.nms

        config_file = cfg.darknet.cfg_file.encode('utf-8')
        weights_file = cfg.darknet.weights_file.encode('utf-8')
        datacfg_file = cfg.darknet.datacfg_file.encode('utf-8')
        self.net = load_net(config_file, weights_file, 0)
        self.meta = load_meta(datacfg_file)


    def do_predict(self, images_string):

        net = self.net
        meta =  self.meta
        thresh = self.thresh
        hier_thresh = self.hier_thresh
        nms = self.nms
        
        predictions = []

        for image_string in images_string:
            predictions.append(detect_on_memory(net, meta, image_string, thresh, hier_thresh, nms))

        return predictions
            