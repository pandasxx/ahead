from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

import os
import sys
import math
sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')
from config.config import cfg


class XceptionPreCore:

    def __init__(self, gpu="0"):

        # read param from config file
        gram_ratio = cfg.xception_pre.gram_ratio
        weights_file = cfg.xception_pre.weights_file
        img_size = cfg.xception_pre.size
        class_num = cfg.xception_pre.class_num
        batch_size = cfg.xception_pre.batch_size

        # set gpu and gram
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = cfg.xception_pre.gram_ratio
        set_session(tf.Session(config=config))

        # build model
        input_tensor = Input((img_size, img_size, 3))
        x = Lambda(xception.preprocess_input)(input_tensor)
        base_model = Xception(input_tensor=x, weights=None, include_top=False)
        m_out = base_model.output
        p_out = GlobalAveragePooling2D()(m_out)
        fc_out = Dropout(1.0)(p_out)
        predictions = Dense(class_num, activation='softmax')(fc_out)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(weights_file)

        # build model done
        self.model = model
        self.batch_size = batch_size

    def do_predict(self, images):
        batch_size = self.batch_size
        model = self.model
        predictions = []
        batches_num = (len(images) / batch_size)

        for i in range(batches_num):
            batch_data = images[i * batch_size: (i + 1) * batch_size]
            predictions.extend(model.predict_on_batch(batch_data))

        return predictions

