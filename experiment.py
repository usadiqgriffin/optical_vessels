from PIL import Image
import numpy as np
from skimage.filters.ridges import sato, frangi
from dataloader import dataloader
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import logging
import matplotlib.pyplot as plt


class Experiment(object):

    def __init__(self, model, data, output_path):

        self.model = model
        self.output_path = output_path
        self.data = data

    #def eval(self):

    def train(self):

        epochs = 200
        save_model_epoch = 20
        batch_size = 30
        internal_shape = self.model.internal_shape
        internal_pixdim = self.model.internal_pixdim

        train_batches =  int(np.ceil(len(self.data.train_set) / batch_size))
        val_batches = int(np.ceil(len(self.data.val_set) / batch_size))

        print('\n**************************************')
        print('len(self.data.train_set) =', len(self.data.train_set))
        print('train_batches =', train_batches)
        print('len(self.data.val_set) =', len(self.data.val_set))
        print('val_batches =', val_batches)
        print('**************************************\n')

        for epoch_index in range(epochs):

            # Training
            random.shuffle(self.data.train_set)
            random.shuffle(self.data.val_set)
            
            training_loss = 0
            validation_loss = 0
            for batch_index in range(train_batches):

                self.model.global_step = self.model.global_step + 1

                start_index = np.minimum(batch_index * batch_size, len(self.data.train_set) - batch_size)

                image = np.zeros(shape=[batch_size, internal_shape[0], internal_shape[1]])
                vessel_mask = np.zeros(shape=[batch_size, internal_shape[0], internal_shape[1]])
                trans_mat = np.zeros(shape=[batch_size, 3, 4])

                for i in range(batch_size):
                    image[i] = self.data.train_set[start_index + i][0]
                    vessel_mask[i] = self.data.train_set[start_index + i][1]

                    # Augmentation
                    flip_z = np.random.random() > 0.5
                    flip_y = np.random.random() > 0.5
                    flip_x = np.random.random() > 0.5

                    # Degrees
                    rotate_z = np.random.random() * 360
                    rotate_y = (np.random.random() * 40) - 20
                    rotate_x = (np.random.random() * 40) - 20

                    # stn_units = voxels * spacing * 2
                    factor = 16  # 1/factor of the frame in all axes
                    translate_z_units = 0
                    translate_y_units = (internal_shape[1] / factor) * internal_pixdim[1] * 2
                    translate_x_units = (internal_shape[0] / factor) * internal_pixdim[0] * 2
                    translate_z = 0
                    translate_y = (np.random.random() * translate_y_units * 2) - translate_y_units
                    translate_x = (np.random.random() * translate_x_units * 2) - translate_x_units

                    trans_mat[i] = np.zeros((3, 4))
                    
                # Training

                #if self.model.internal_shape != image.shape[1:]:
                #logging.info(f"Inputs shapes, expected:{self.model.internal_shape}, found:{image.shape[1:]}")
                _, loss, summary_loss, summary_loss0 = self.model.sess.run(
                    [self.model.optimizer, 
                        self.model.loss,
                        self.model.loss_summary_node,
                        self.model.loss0_summary_node],
                    feed_dict={
                        self.model.image: image,
                        self.model.vessel_mask: vessel_mask,
                        self.model.trans_mat: trans_mat,
                        self.model.training: 1})

                print('Epoch: ' + str(epoch_index) + '/' + str(epochs) + ', Batch: ' + str(batch_index) + '/' + str(train_batches) + ', Train Loss: ' + str(loss))
                logging.debug(f"Image DR:({np.min(image)}-{np.max(image)}),Label DR:({np.min(vessel_mask)}-{np.max(vessel_mask)})")

                training_loss += loss
                # Tensorboard
                if batch_index == 0:
                    self.model.summary_writer_train.add_summary(summary_loss, self.model.global_step)
                    self.model.summary_writer_train.add_summary(summary_loss0, self.model.global_step)
                    self.model.summary_writer_train.flush()

            # Saving the model to file
            if (epoch_index % save_model_epoch == 0) or (epoch_index == epochs - 1):
                print('Saving model for epoch ' + str(epoch_index) + ' / global step ' + str(self.model.global_step))
                self.model.saver.save(self.model.sess, self.output_path + '/models/model', self.model.global_step)
                val_outputs = self.eval(load_mode=1)  # Evaluating the model
                random.shuffle(val_outputs)
                val_outputs = val_outputs[:5]

                for i, output in enumerate(val_outputs):
                    image_trans, vessel_mask_trans, pred_vessels, _ = output
                    images = [image_trans, vessel_mask_trans, pred_vessels]

                    logging.debug(f"Images:{len(images)}, image shape:{images[0].shape}")
                    image_name = ['orig', 'true_vessels', "pred_vessels"]
                    for img_i, _ in enumerate(images):
                        
                        #im = Image.fromarray((np.squeeze(images[img_i]) * 255.0).astype(np.uint8))
                        output_file_path = self.output_path + "/epoch_" + str(epoch_index) + "_image_" + str(i) + "_" + image_name[img_i] + ".jpeg"
                        plt.imsave(output_file_path, np.squeeze(images[img_i]), cmap=plt.cm.gray)
                        #im.save( output_file_path)

                logging.debug(f"Saving validation images at {output_file_path}")

            # Log epoch-wise training and validation losses

            #Solution 1: Didn't work
            '''training_loss_ = tf.placeholder(tf.float32, [])
            summary_op = tf.scalar_summary("value_log", training_loss_)

            with tf.Session() as sess:
                sess.run(summary_op, feed_dict={training_loss__: tf.math.reduce_mean(training_loss)})'''

            #Solution 2: Didn't work
            '''summary_op = tf.summary.tensor_summary('training_loss', tf.math.reduce_mean(training_loss))
            # Create the summary
            summary_str = tf.Session().run(summary_op)'''

            #Solution 3: Working
            #logging.debug(f"Epoch Training Loss:{training_loss/train_batches}")
            '''summary = tf.Summary(value=[
                tf.Summary.Value(tag="summary_tag", simple_value=training_loss/train_batches), 
            ])
            self.model.summary_writer_train.add_summary(summary, global_step=epoch_index)
            self.model.summary_writer_train.flush()'''
            
            # Validation
            if (epoch_index % 1 == 0) or (epoch_index == epochs - 1):

                print('Validating Epoch ' + str(epoch_index) + '/' + str(epochs) + ' ...')
                avg_loss = 0

                for batch_index in range(val_batches):

                    start_index = np.minimum(batch_index * batch_size, len(self.data.val_set) - batch_size)

                    image = np.zeros(shape=[batch_size, internal_shape[0], internal_shape[1]])
                    vessel_mask = np.zeros(shape=[batch_size, internal_shape[0], internal_shape[1]])
                    trans_mat = np.zeros(shape=[batch_size, 3, 4])

                    for i in range(batch_size):
                        image[i] = self.data.val_set[start_index + i][0]
                        vessel_mask[i] = self.data.val_set[start_index + i][1]
                        
                        trans_mat[i] = np.zeros((3, 4))

                    # Validation
                    loss, summary_loss, summary_loss0 = self.model.sess.run(
                        [self.model.loss,
                            self.model.loss_summary_node,
                            self.model.loss0_summary_node],
                        feed_dict={
                            self.model.image: image,
                            self.model.vessel_mask: vessel_mask,
                            self.model.trans_mat: trans_mat,
                            self.model.training: 0})

                    avg_loss = avg_loss + loss

                    # Tensorboard
                    if batch_index == 0:
                        self.model.summary_writer_val.add_summary(summary_loss, self.model.global_step + batch_index)
                        self.model.summary_writer_val.add_summary(summary_loss0, self.model.global_step + batch_index)

                avg_loss = avg_loss / val_batches

                '''summary = tf.Summary(value=[
                    tf.Summary.Value(tag="summary_tag", simple_value=avg_loss), 
                ])
                self.model.summary_writer_val.add_summary(summary, global_step=epoch_index)
                self.model.summary_writer_val.flush()'''
                print('Epoch: ' + str(epoch_index) + '/' + str(epochs) + ', Validation Loss: ' + str(avg_loss))


    def eval(self, load_mode):
        
        if load_mode == 1:
            # Forcing the model to run on the validation set only
            datasets = [[], self.data.val_set, []]
        else:
            # Running the model on all the train/val/test set, depending on what is loaded to memory (i.e. load_mode)
            datasets = [self.data.train_set, self.data.val_set, self.data.test_set]

        model_outputs = []  # will contain everything needed to run the test and generate the report
        loss0_arr = []

        for dataset in datasets:

            for i in range(len(dataset)):

                print(i, '/', len(dataset))

                # Validating the validation set
                image = np.expand_dims(dataset[i][0], 0)
                vessel_mask = np.expand_dims(dataset[i][1], 0)
                row = ''
                #pred_vessel_internal = dataset[i][3]


                trans_mat = np.expand_dims(np.zeros(shape=(3, 4)), 0)

                image_trans, vessel_mask_trans, pred_vessels, loss0 = self.model.sess.run(
                    [self.model.image_trans,
                     self.model.vessel_mask_trans,
                     self.model.pred_vessels,
                     self.model.loss0],
                    feed_dict={
                        self.model.image: image,
                        self.model.vessel_mask: vessel_mask,
                        self.model.trans_mat: trans_mat,
                        self.model.training: 0})

                model_outputs.append([image_trans,
                                      vessel_mask_trans,
                                      pred_vessels,
                                      row])
                loss0_arr.append(loss0)
        
        return model_outputs