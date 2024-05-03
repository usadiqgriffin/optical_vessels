#import lightning as L

import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import logging
from tensorflow.keras.losses import BinaryCrossentropy

class UNetModel(object):

    def __init__(self, output_path, clean_output, create_summary, gpu = 0, losstype = ""):
        self.output_path = output_path
        self.losstype = losstype
        self.seg_thresh = 0.5

        #os.environ["CUDA_VISIBLE_DEVICES"] = ''#+str(gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # Defining the Tensorflow graph
        self.sess = tf.Session()

        if clean_output:
            # Cleaning the output folder
            if os.path.isdir(output_path):
                now = datetime.now()
                now_str = now.strftime("%Y%m%d_%H%M%S")
                os.system('mv ' + output_path + ' ' + output_path + '_' + now_str)


        self.summary_writer_train = tf.summary.FileWriter(output_path + '/tensorboard/train', graph=self.sess.graph)
        self.summary_writer_val = tf.summary.FileWriter(output_path + '/tensorboard/val', graph=self.sess.graph)


    def model_summary(self):
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    def unet(self, input):

        down1 = tf.layers.conv2d(input, 8, (3, 3), (1, 1), 'same', activation=tf.nn.relu, name='conv2d_1_1')  # [batch, 80, 192, 160] -> [256,256,32]
        down1 = tf.layers.conv2d(down1, 8, (3, 3), (1, 1), 'same',  activation=tf.nn.relu, name='conv2d_1_2')
        down1 = tf.layers.batch_normalization(down1, training=self.training)

        down2 = tf.layers.conv2d(down1, 8, (3, 3), (2, 2), 'same', activation=tf.nn.relu)  # [40, 96, 80] -> [128,128,16]
        down2 = tf.layers.conv2d(down2, 8, (3, 3), (1, 1), 'same', activation=tf.nn.relu)
        down2 = tf.layers.batch_normalization(down2, training=self.training)

        down3 = tf.layers.conv2d(down2, 16, (3, 3), (2, 2), 'same', activation=tf.nn.relu)  # [20, 48, 40] -> [64,64,8]
        down3 = tf.layers.conv2d(down3, 16, (3, 3), (1, 1), 'same', activation=tf.nn.relu)
        down3 = tf.layers.batch_normalization(down3, training=self.training)

        down4 = tf.layers.conv2d(down3, 16, (3, 3), (2, 2), 'same', activation=tf.nn.relu)  # [10, 24, 20] -> [32,32,4]
        down4 = tf.layers.conv2d(down4, 16, (3, 3), (1, 1), 'same', activation=tf.nn.relu)
        down4 = tf.layers.batch_normalization(down4, training=self.training)

        latent = tf.layers.conv2d(down4, 32, (3, 3), (2, 2), 'same', activation=tf.nn.relu)  # [5, 12, 10] -> [16, 16, 2]
        latent = tf.layers.conv2d(latent, 32, (3, 3), (1, 1), 'same', activation=tf.nn.relu)
        latent = tf.layers.batch_normalization(latent, training=self.training)

        up4 = tf.layers.conv2d_transpose(latent, 16, (3, 3), (2, 2), 'same', activation=tf.nn.relu)  # [batch, 10, 24, 20] -> [16,32,4]
        up4 = tf.concat([up4, down4], axis=-1)
        up4 = tf.layers.conv2d(up4, 16, (3, 3), (1, 1), 'same', activation=tf.nn.relu)
        up4 = tf.layers.batch_normalization(up4, training=self.training)

        up3 = tf.layers.conv2d_transpose(up4, 16, (3, 3), (2, 2), 'same', activation=tf.nn.relu)  # [batch, 20, 48, 40]
        up3 = tf.concat([up3, down3], axis=-1)
        up3 = tf.layers.conv2d(up3, 16, (3, 3), (1, 1), 'same', activation=tf.nn.relu)
        up3 = tf.layers.batch_normalization(up3, training=self.training)

        up2 = tf.layers.conv2d_transpose(up3, 8, (3, 3), (2, 2), 'same', activation=tf.nn.relu)  # [batch, 40, 96, 80]
        up2 = tf.concat([up2, down2], axis=-1)
        up2 = tf.layers.conv2d(up2, 8, (3, 3), (1, 1), 'same', activation=tf.nn.relu)
        up2 = tf.layers.batch_normalization(up2, training=self.training)

        up1 = tf.layers.conv2d_transpose(up2, 8, (3, 3), (2, 2), 'same')  # [batch, 80, 192, 160]
        up1 = tf.concat([up1, down1], axis=-1)
        up1 = tf.layers.conv2d(up1, 8, (3, 3), (1, 1), 'same', activation=tf.nn.relu)
        up1 = tf.layers.batch_normalization(up1, training=self.training)

        logits = tf.layers.conv2d(up1, self.maps_count, (3, 3), (1, 1), 'same', activation=None)

        return logits


    def dice_loss(self, true, pred):


        #tf.compat.v1.Print(true.shape, [true.shape], message="\nTrue shapes: ", summarize=80)
        
        smooth = 1e-6
        true = tf.cast(true, tf.float32)
        pred = tf.cast(pred, tf.float32)
        
        numerator = tf.reduce_sum(true * pred, axis=[1, 2, 3]) + smooth
        denominator = tf.reduce_sum(true, axis=[1, 2, 3]) + tf.reduce_sum(pred, axis=[1, 2, 3]) + smooth
        clot_mask_sum = tf.reduce_mean(tf.reduce_sum(true, axis=[1, 2, 3]))
        loss = -(numerator / denominator)

        return loss, numerator, denominator, clot_mask_sum

    def bce_loss(self, true, pred):
        true = tf.cast(true, tf.float32)
        pred = tf.cast(pred, tf.float32)
        bce = BinaryCrossentropy(from_logits=True)(true, pred)

        return bce
    
    def define_model(self, trainable=True):

        self.memory_shape = np.array([512, 512])
        self.memory_pixdim = np.array([1.0, 1.0])

        self.internal_shape = np.array([512, 512])
        self.internal_pixdim = np.array([1.0, 1.0])
        
        ''' 
        self.internal_shape = np.array([128, 256, 64])
        self.internal_pixdim = np.array([2.0, 1.0, 2.0])'''

        self.maps_count = 1  # image, clot_location_map, tmax, cbf, cbv
        self.max_image_val = 1.0 
        self.max_tmax_val = 1.0

        with tf.variable_scope('Input'):

            self.image = tf.placeholder(tf.float32, shape=(None, None, None), name='image')
            self.clot_mask = tf.placeholder(tf.float32, shape=(None, None, None), name='clot_mask')

            self.trans_mat = tf.placeholder(tf.float32, shape=(None, 3, 4), name='trans_mat')  # [batch, 3, 4]
            self.training = tf.placeholder(tf.bool, name='training')

        with tf.variable_scope('Preprocessing'):
            self.batch = tf.shape(self.image)[0]
            print('Not augmenting')
            self.image_trans = tf.expand_dims(self.image, -1)
            self.image_trans = self.image_trans - tf.math.reduce_min(self.image_trans)/(tf.math.reduce_max(self.image_trans)- tf.math.reduce_min(self.image_trans))
            
            self.clot_mask_trans = tf.expand_dims(self.clot_mask, -1)
            self.clot_mask_trans = tf.cast(self.clot_mask_trans > 0, tf.float32)
            #self.clot_mask_trans = self.clot_mask_trans - tf.math.reduce_min(self.clot_mask_trans)/(tf.math.reduce_max(self.clot_mask_trans)- tf.math.reduce_min(self.clot_mask_trans))

            print('max image val:{}'.format(self.max_image_val))

        with tf.variable_scope('classifier'): 

            self.logits = self.unet(self.image_trans)   # [batch, 512, 512, 1], single activation for class 1
            
            #self.logits = self.logits[:, :, :, 0]   # [batch, 512, 512]
            self.sigmoid = tf.math.sigmoid(self.logits[:, :, :, 0], name='sigmoid')  # [batch, 512, 512]

            self.softmax = tf.nn.softmax(self.logits[:, :, :, 0], name='softmax_cls')  # [batch, 512, 512, 2]
            self.pred = tf.argmax(self.softmax, axis=-1)  # [batch, 512, 512]

            self.pred_clot = tf.expand_dims(tf.identity(tf.cast(self.sigmoid > self.seg_thresh, tf.float32), name='pred_clot'), -1)  # [batch, 96, 256, 256]

            logging.debug(f"SHAPES:logits shape:{self.logits.shape}, sigmoid shape:{self.sigmoid.shape}, softmax shape:{self.softmax.shape}, pred shape:{self.pred_clot.shape}")

            '''if self.training:
                logging.debug(f"Shapes: logits:{self.logits.shape}, sigmoid:{self.sigmoid.shape}, pred_clot:{self.pred_clot.shape}")'''

            

        with tf.variable_scope('loss'):

            # CTP maps MSE loss
            
            if self.losstype == 'mse':
                self.loss0 = (self.clot_mask_trans - self.logits) ** 2  # [batch, 80, 192, 160, 1]
                #self.loss0 = tf.reduce_sum(self.loss0, [1, 2, 3, 4]) / 2  # [batch]
                #print('sse loss:{}, N={}, avg={}'.format(tf.reduce_sum(self.loss0),tf.shape(self.loss0),tf.reduce_mean(self.loss0)))
                
                #self.N = tf.reduce_prod(tf.shape(self.loss0))
                self.sse = format(tf.reduce_sum(self.loss0))
                self.loss0 = tf.reduce_mean(self.loss0)  # [batch]'''
            else:
                
                # Use dice loss instead
                self.loss0  = self.bce_loss(self.clot_mask_trans, self.logits)
                #self.loss0, self.numerator0, self.denominator0, self.clot_mask_sum0  = self.dice_loss(self.clot_mask_trans, self.logits)
                self.loss0 = tf.reduce_mean(self.loss0)

                
            self.loss0_summary_node = tf.summary.scalar('loss0', self.loss0)

            # Total loss
            self.loss = self.loss0
            self.loss_summary_node = tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('optimizer'):

            # Batch Normalization
            # Ensures that we execute the update_ops() before performing the train_step
            # This updates the estimated population statistics during training, which is later used during testing
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

        with tf.variable_scope('Output'):
            self.pred_vessel = tf.sigmoid(self.logits)

    def initialize_weights(self, global_step):

        self.saver = tf.train.Saver(max_to_keep=None)
        init = tf.global_variables_initializer()

        if global_step == 0:
            self.sess.run(init)
        else:
            if global_step == -1 :
                ckpt_list = glob.glob(self.output_path + '/models/model-*.meta')
                epoch_list = []
                for ckpt in ckpt_list:
                    epoch_list.append(int(ckpt.split('/')[-1].split('.')[0].split('-')[-1]))
                epoch_list = sorted(epoch_list)
                global_step = epoch_list[-1]

            mdl_path = self.output_path + '/models/model-' + str(global_step)
            print('\n********************************')
            print('Loading model-' + str(global_step) + ',saved at ' + mdl_path)
            print('********************************\n')
            self.saver.restore(self.sess, mdl_path)

        self.global_step = global_step

