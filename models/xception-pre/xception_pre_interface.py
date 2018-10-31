from xception_pre import XceptionPre
from config.config import cfg
import numpy as np

class XceptionPreInterface:

        def __init__(self,  gpu="0"):

                self.model = XceptionPreCore(gpu)
                self.thresh = cfg.xception_pre.thresh
                self.classes = cfg.xception_pre.classes
                self.result = cfg.xception_pre.result
                self.is_normal = cfg.xception_pre.is_normal
                self.is_abnormal = cfg.xception_pre.is_abnormal

########### input ################
#       list:                   
#               dict 1216
#                       x               : top left corner coordinate x
#                       y               : top left corner coordinate y
#                       w               : width of the image
#                       h               : height of the image
#                       img_data        : numpy image data 1216
#                       img_pre         : numpy image data 299
#
#       image_data_key
#                       str             img_pre
#       result_key
#                       str             is_normal

########### output ################
#       list:
#               dict 1216 after pre
#                       x                       : top left corner coordinate x
#                       y                       : top left corner coordinate y
#                       w                      : width of the image
#                       h                       : height of the image                                                   
#                       img_data          : numpy image data 1216
#                       img_pre            : numpy image data 299
#                       is_normal          : IS_NORAML or IS_ABNORAML
        def predict(image_dict_list, image_data_key, result_key):

                model = self.model
                thresh = self.thresh
                classes = self.classes
                result = self.result
                IS_NORAML = self.is_normal
                IS_ABNORAML = self.is_abnormal

                image_list = []
                for image_dict in image_dict_list:
                       image_list.append(image_dict[image_data_key])

                predictions = model.do_predict(image_list)

                for i, image_dict in enumerate(image_dict_list):

                        max_prohibtion_index = np.argmax(predictions[i])
                        
                        if ((result[max_prohibtion_index] == IS_NORAML) and ( predictions[i][max_prohibtion_index] >=  thresh)):
                                image_dict_list[result_key] = IS_NORAML
                        else:
                                image_dict_list[result_key] = IS_ABNORAML