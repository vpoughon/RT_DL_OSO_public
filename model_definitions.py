from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Concatenate, core, Dropout, concatenate, Cropping2D, Convolution2D, ConvLSTM2D, Conv3D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import TimeDistributed, LSTM, core, Dropout, concatenate, Cropping2D, Dense, Flatten
from keras import backend as K

#### For multi GPU
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model

import tensorflow as tf

def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))
            
        return Model(input=model.inputs, output=merged)

#####################

def sobel_categorical_crossentropy(weights):
    #this contains both X and Y sobel filters in the format (3,3,1,2)
    #size is 3 x 3, it considers 1 input channel and has two output channels: X and Y (for dX and dY)
    sobelFilter = K.variable([[[[1.,  1.]], [[0.,  2.]],[[-1.,  1.]]],
                          [[[2.,  0.]], [[0.,  0.]],[[-2.,  0.]]],
                          [[[1., -1.]], [[0., -2.]],[[-1., -1.]]]])
    weights = K.variable(weights)
    
    def expandedSobel(inputTensor):
        #this considers data_format = 'channels_last'
        inputChannels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
        #inputChannels = K.reshape(K.ones_like(inputTensor[0,:,0,0]),(1,1,-1,1))# TODO:OLD VERSION
        #if you're using 'channels_first', use inputTensor[0,:,0,0] above
        return sobelFilter * inputChannels
    
    def sobelLoss(yTrue,yPred):
        #get the sobel filter repeated for each input channel
        filt1 = expandedSobel(yTrue)
        filt2 = expandedSobel(yPred)

        #calculate the sobel filters for yTrue and yPred
        #this generates twice the number of input channels 
        #a X and Y channel for each input channel
        sobelTrue = K.depthwise_conv2d(yTrue,filt1, data_format="channels_last")
        sobelPred = K.depthwise_conv2d(yPred,filt2, data_format="channels_last")
        #sobelTrue = K.depthwise_conv2d(yTrue,filt1)# TODO:OLD VERSION
        #sobelPred = K.depthwise_conv2d(yPred,filt2)# TODO:OLD VERSION
        sobelPred = K.sum(sobelPred,axis=3)
        sobelTrue = K.sum(sobelTrue,axis=3)
        #sobelPred = K.sum(sobelPred,axis=1)# TODO:OLD VERSION
        #sobelTrue = K.sum(sobelTrue,axis=1)# TODO:OLD VERSION

        #now you just apply the mse:
        return K.mean(K.square(sobelTrue - sobelPred))/50590
    
    def loss_temp(y_true, y_pred):
        #y_pred = K.reshape(y_pred,(16,17,64,64)) # TODO:OLD VERSION
        y_pred = K.reshape(y_pred,(16,16,64,64))
        y_true = K.reshape(y_true,(16,17,64,64))
        #y_pred_classes = y_pred[:,:-1,:,:] / K.sum(y_pred[:,:-1,:,:], axis=-1, keepdims=True)# TODO:OLD VERSION
        y_pred_classes = y_pred / K.sum(y_pred, axis=-1, keepdims=True)
        y_pred_classes = K.clip(y_pred_classes, K.epsilon(), 1 - K.epsilon())
        y_true_contours = K.reshape(y_true[:,-1,:,:],(16,1,64,64))
        loss = y_true[:,:-1,:,:] * K.log(y_pred_classes) #* weights[:-1]
        #loss = -K.sum(loss, -1) + weights[-1]*sobelLoss(y_true_contours,y_pred[:,:-1,:,:])# TODO:OLD VERSION
        loss = -K.sum(loss, -1) #+ weights[-1]*sobelLoss(y_true_contours,y_pred)
        return loss
    def loss(y_true, y_pred):
        y_pred = K.reshape(y_pred,(16,64,64,16))
        y_true = K.reshape(y_true,(16,64,64,17))
        #y_pred_classes = y_pred[:,:-1,:,:] / K.sum(y_pred[:,:-1,:,:], axis=-1, keepdims=True)# TODO:OLD VERSION
        y_pred_classes = y_pred / K.sum(y_pred, axis=-1, keepdims=True)
        y_pred_classes = K.clip(y_pred_classes, K.epsilon(), 1 - K.epsilon())
        loss = y_true[...,:-1] * K.log(y_pred_classes) #* weights[:-1]
        #loss = -K.sum(loss, -1) + weights[-1]*sobelLoss(y_true_contours,y_pred[:,:-1,:,:])# TODO:OLD VERSION
        y_true_contours = K.reshape(y_true[...,-1],(16,64,64,1))
        loss = -K.sum(loss, -1) + weights[-1]*sobelLoss(y_true_contours,y_pred)
        return loss
    return loss

def weighted_semisupervised_loss(weights):
    weights = K.variable(weights)

    def loss(loss_inputs):
        y_pred = loss_inputs[0]
        y_true = loss_inputs[2]
        x_input = loss_inputs[1]
        x_pred = loss_inputs[3]

        # scale predictions so that the class probas of each sample sum to 1
        y_is_labeled = K.sum(y_pred) #is 1 if the sample is labeled, 0 if not

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_is_labeled * y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)

        loss += K.sqrt(K.sum(K.square(x_input - x_output)))
        return loss

    return loss


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

def acc_labeled_only(y_true, y_pred):
    '''
    accuracy evaluated only on labeled pixels
    '''
    annot_mask = K.cast(K.not_equal(K.max(y_true, axis=-1), 0), 'int64')
    num_annot = K.sum(annot_mask)
    num_not_annot = K.sum(1 - annot_mask)
    y_true_label = K.argmax(y_true, axis=-1)
    y_pred_label = K.argmax(y_pred, axis=-1)

    # all not annotated GT will be 0
    y_pred_all = (y_pred_label + K.ones_like(y_pred_label)) * annot_mask
    # all not annotated PRED will be 0, first valid class=1
    y_true_all = (y_true_label + K.ones_like(y_true_label)) * annot_mask

    # 0 = classes do not match, else 1. will be 1 for non annotated pixels
    mask_class_ok = K.cast(K.equal(y_pred_all, y_true_all), 'int64')

    # count all matching, subtract non annotated, divide by number of annotated
    return K.cast(K.sum(mask_class_ok) - num_not_annot, 'float32') / K.cast(num_annot, 'float32')


def fscore_class(u_class):
    '''
    metric computing the FScore of each class
    Problem : returns 0 if the class is not in the patch, and this value 0 is taken into account in the mean of the different patches, so results are lower than expected. 
    ==> Fscore for each class not computable with a metric, we use a callback instead
    @param u_class : between 0 and class_number
    '''
    def fscore_by_class(y_true, y_pred):
        '''
        accuracy evaluated only on labeled pixels
        '''
        annot_mask = K.cast(K.not_equal(K.max(y_true, axis=-1), 0), 'int64')
        num_annot = K.sum(annot_mask)
        num_not_annot = K.sum(1 - annot_mask)
        y_true_label = K.argmax(y_true, axis=-1) # between 0 and nb_class -1
        y_pred_label = K.argmax(y_pred, axis=-1) # between 0 and nb_class -1
        #for u_class in range(1, K.int_shape(y_pred)[-1] + 1):
        # all not annotated GT will be 0 
        y_pred_all = (y_pred_label + K.ones_like(y_pred_label)) * annot_mask # between 0 (not labeled) and nb_class
        # all not annotated PRED will be 0, first valid class=1
        y_true_all = (y_true_label + K.ones_like(y_true_label)) * annot_mask # between 0 (not labeled) and nb_class
        
        # for the class considered
        y_pred_all_class = K.cast(K.equal(y_pred_all, u_class * K.ones_like(y_pred_all)), 'int64') # 1 where y_pred_all == class else 0
        y_true_all_class = K.cast(K.equal(y_true_all, u_class * K.ones_like(y_true_all)), 'int64') # 1 where y_true_all == class else 0

        # 0 = classes do not match, else 1. will be 1 for non annotated pixels
        t_tp = K.cast(K.equal(y_pred_all_class, y_true_all_class), 'int64') # true positive tensor
        t_tp_mask = K.cast(K.not_equal(y_pred_all_class, 0), 'int64')
        t_tp = t_tp * t_tp_mask # to keep only y_pred_all_class = y_true_all_class = 1
        f_tp = K.cast(K.sum(t_tp), 'float32') # number of true positive
        f_tp_plus_fp = K.cast(K.sum(y_pred_all_class), 'float32')
        f_tp_plus_fn = K.cast(K.sum(y_true_all_class), 'float32')
        f_prec = K.switch(K.not_equal(f_tp_plus_fp, K.zeros_like(f_tp_plus_fp)), f_tp / f_tp_plus_fp, K.zeros_like(f_tp_plus_fp))
        f_recall = K.switch(K.not_equal(f_tp_plus_fn, K.zeros_like(f_tp_plus_fn)), f_tp / f_tp_plus_fn, K.zeros_like(f_tp_plus_fn))
        f_Fscore = K.switch(K.not_equal(f_recall + f_prec, K.zeros_like(f_recall)), 2 * f_recall * f_prec / (f_recall + f_prec), K.zeros_like(f_recall))
        # count all matching, subtract non annotated, divide by number of annotated
        return K.cast(f_Fscore, 'float32')
    u_class += 1 # classes between 1 and class_number
    return fscore_by_class




def get_unet_weights(n_ch, n_output, patch_height, patch_width, weights):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    #
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    #
    conv6 = Conv2D(n_output, (1, 1), activation='relu',padding='same')(conv5)
    
    # !!!! KEEP THIS RESHAPE AND PERMUTE FOR SOFTMAX !!!!
    conv6 = core.Reshape((n_output,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=False)
    #model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    loss = weighted_categorical_crossentropy(weights)
    model.compile(optimizer="Adam", loss=loss, metrics=['accuracy'])
    
    return model

def get_unet_mlp_ts_highdropout(n_ch, n_timesteps, n_output, patch_height, patch_width,hidden_layers, optimizer="Adam", weights = None):
# create u-net model
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.3)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = Dropout(0.3)(conv1) # new dropout
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = Dropout(0.3)(conv2) # new dropout
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Dropout(0.3)(conv3) # new dropout

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Dropout(0.3)(conv4) # new dropout
    #
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Dropout(0.3)(conv5) # new dropout
    unet_model = Model(inputs=inputs, outputs=conv5)
    #
    
    ###
    
    l_prev = Conv2D(hidden_layers[0], (1, 1), activation='relu', padding='same')(inputs)
    l_prev = Dropout(0.3)(l_prev)
    for n_hidden_neurons in hidden_layers[1:]:
        fc_new = Conv2D(n_hidden_neurons, (1, 1), activation='relu', padding='same')(l_prev)
        # added by VP to decrease overfitting:
        fc_new = Dropout(0.3)(fc_new)
        l_prev = fc_new
    mlp_model = Model(inputs=inputs, outputs=l_prev)
    
    # share this model among temporal samples
    inputs_list = []
    out_list = []
    out_mlp_list = []
    for i in range(n_timesteps):
        input_patch = Input((n_ch, patch_height, patch_width))
        inputs_list.append(input_patch)
        out_unet_layer = unet_model(input_patch)
        out_mlp_layer = mlp_model(input_patch)
        out_list.append(out_unet_layer)
        out_mlp_list.append(out_mlp_layer)
    out_unet = concatenate(out_list,axis=1)
    out_mlp = concatenate(out_mlp_list,axis=1)
    
    out_concat = concatenate([out_mlp, out_unet],axis=1)
    #print(K.shape(out_concat))
    out_concat = core.Reshape(((hidden_layers[-1]+32)*n_timesteps,patch_height,patch_width))(out_concat)
    
    # create common layers for softmax
    out2 = Conv2D(n_output, (1, 1), activation='relu',padding='same')(out_concat)
    # !!!! KEEP THIS RESHAPE AND PERMUTE FOR SOFTMAX !!!!
    out2 = core.Reshape((n_output,patch_height*patch_width))(out2)
    out2 = core.Permute((2,1))(out2)
    ############
    out2 = core.Activation('softmax')(out2)
    
    
    # get unified model
    model_out = Model(inputs_list,out2)
    #mlp_unet_loss = mlp_unet_loss()


    if weights is not None:
        loss = weighted_categorical_crossentropy(weights)
        model_out.compile(optimizer=optimizer, loss=loss, metrics=[acc_labeled_only])
    else:
        model_out.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc_labeled_only])
    return model_out


def get_unet_mlp_ts(n_ch, n_timesteps, n_output, patch_height, patch_width,hidden_layers, optimizer="Adam", weights = None):
# create u-net model
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    #
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    unet_model = Model(inputs=inputs, outputs=conv5)
    #
    
    ###
    
    l_prev = Conv2D(hidden_layers[0], (1, 1), activation='relu', padding='same')(inputs)
    for n_hidden_neurons in hidden_layers[1:]:
        fc_new = Conv2D(n_hidden_neurons, (1, 1), activation='relu', padding='same')(l_prev)
        # added by VP to decrease overfitting:
        fc_new = Dropout(0.2)(fc_new)
        l_prev = fc_new
    mlp_model = Model(inputs=inputs, outputs=l_prev)
    
    # share this model among temporal samples
    inputs_list = []
    out_list = []
    out_mlp_list = []
    for i in range(n_timesteps):
        input_patch = Input((n_ch, patch_height, patch_width))
        inputs_list.append(input_patch)
        out_unet_layer = unet_model(input_patch)
        out_mlp_layer = mlp_model(input_patch)
        out_list.append(out_unet_layer)
        out_mlp_list.append(out_mlp_layer)
    out_unet = concatenate(out_list,axis=1)
    out_mlp = concatenate(out_mlp_list,axis=1)
    
    out_concat = concatenate([out_mlp, out_unet],axis=1)
    #print(K.shape(out_concat))
    out_concat = core.Reshape(((hidden_layers[-1]+32)*n_timesteps,patch_height,patch_width))(out_concat)
    
    # create common layers for softmax
    out2 = Conv2D(n_output, (1, 1), activation='relu',padding='same')(out_concat)
    # !!!! KEEP THIS RESHAPE AND PERMUTE FOR SOFTMAX !!!!
    out2 = core.Reshape((n_output,patch_height*patch_width))(out2)
    out2 = core.Permute((2,1))(out2)
    ############
    out2 = core.Activation('softmax')(out2)
    
    
    # get unified model
    model_out = Model(inputs_list,out2)
    #mlp_unet_loss = mlp_unet_loss()

    if weights is not None:
        loss = weighted_categorical_crossentropy(weights)
        model_out.compile(optimizer=optimizer, loss=loss, metrics=[acc_labeled_only]) 
    else:
        model_out.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc_labeled_only]) 
    return model_out

def get_unet_only(n_ch, n_timesteps, n_output, patch_height, patch_width, optimizer="Adam", weights = None):
    '''
    Total params: 485,297
    '''
# create u-net model
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    #
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    unet_model = Model(inputs=inputs, outputs=conv5)
    #

    # share this model among temporal samples
    inputs_list = []
    out_list = []
    #out_mlp_list = []
    for i in range(n_timesteps):
        input_patch = Input((n_ch, patch_height, patch_width))
        inputs_list.append(input_patch)
        out_unet_layer = unet_model(input_patch)
        out_list.append(out_unet_layer)
    out_unet = concatenate(out_list,axis=1)

    # create common layers for softmax
    out2 = Conv2D(n_output, (1, 1), activation='relu',padding='same')(out_unet)
    # !!!! KEEP THIS RESHAPE AND PERMUTE FOR SOFTMAX !!!!
    out2 = core.Reshape((n_output,patch_height*patch_width))(out2)
    out2 = core.Permute((2,1))(out2)
    ############
    out2 = core.Activation('softmax')(out2)
    
    
    # get unified model
    model_out = Model(inputs_list,out2)

    if weights is not None:
        loss = weighted_categorical_crossentropy(weights)
        model_out.compile(optimizer=optimizer, loss=loss, metrics=[acc_labeled_only])
    else:
        model_out.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc_labeled_only])
    return model_out
    


def get_unetsmall_mlp_convlstm(n_ch, n_timesteps, n_output, patch_height, patch_width, hidden_layers, optimizer="Adam", weights=None):
    '''
    Unet only 2 level deep
    '''
    # create u-net model
    inputs = Input((n_timesteps, n_ch, patch_height, patch_width))
    conv1 = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs) # (None, 11, 32, 64, 64)
    #conv1bis = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs) 
    conv1 = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=True)(conv1) # (None, 11, 32, 64, 64)
    #print(K.int_shape(conv1))
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    conv2 = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(pool1)
    conv2 = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=True)(conv2)
    #pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    #
    #conv3 = ConvLSTM2D(128, (3, 3), activation='relu', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(pool2)
    #conv3 = ConvLSTM2D(128, (3, 3), activation='relu', padding='same', return_sequences=True)(conv3)

    #up1 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv3), conv2], axis=-3)
    #conv4 = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(up1)
    #conv4 = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=True)(conv4)
    #
    up2 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv2), conv1], axis=-3)
    conv5 = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(up2)
    conv5 = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=False)(conv5)
    #unet_model = Model(inputs=inputs, outputs=conv5)
    ##
    l_prev = ConvLSTM2D(hidden_layers[0], (1, 1), activation='relu', padding='same', return_sequences=True)(inputs)
    for n_hidden_neurons in hidden_layers[1:-1]:
        fc_new = ConvLSTM2D(n_hidden_neurons, (1, 1), activation='relu', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(l_prev)
        # added by VP to decrease overfitting:
        #fc_new = Dropout(0.2)(fc_new)
        l_prev = fc_new
    fc_new = ConvLSTM2D(hidden_layers[-1], (1, 1), activation='relu', padding='same', return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(l_prev)
    #fc_new = Dropout(0.2)(fc_new)
    l_prev = fc_new
    #mlp_model = Model(inputs=inputs, outputs=l_prev)
    out_concat = concatenate([l_prev, conv5], axis=-3)
    # create common layers for softmax
    out2 = Conv2D(n_output, (1, 1), activation='relu',padding='same')(out_concat) # (None, 82, 64, 64)
    # !!!! KEEP THIS RESHAPE AND PERMUTE FOR SOFTMAX !!!!
    out2 = core.Reshape((n_output, patch_height*patch_width))(out2)
    out2 = core.Permute((2,1))(out2)
    ############
    out2 = core.Activation('softmax')(out2)
    
    
    # get unified model
    model_out = Model(inputs, out2)
    
    if weights is not None:
        loss = weighted_categorical_crossentropy(weights)
        model_out.compile(optimizer=optimizer, loss=loss, metrics=[acc_labeled_only]) #+ [fscore_class(u_class) for u_class in range(n_output)])
    else:
        model_out.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc_labeled_only]) # + [fscore_class(u_class) for u_class in range(n_output)])
    return model_out
    
def get_unet_mlp_convlstm_full(n_ch, n_timesteps, n_output, patch_height, patch_width, hidden_layers, optimizer="Adam", weights=None):
    '''
    Replacement of all Conv2D (of get_unet_mlp_ts model) by ConvLSTM2D
    '''
    # create u-net model
    inputs = Input((n_timesteps, n_ch, patch_height, patch_width))
    conv1 = ConvLSTM2D(32, (3, 3), activation='tanh', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs) # (None, 11, 32, 64, 64)
    #conv1bis = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs) ==> seems similar to ConvLSTM2D
    conv1 = ConvLSTM2D(32, (3, 3), activation='tanh', padding='same', return_sequences=True)(conv1) # (None, 11, 32, 64, 64)
    #print(K.int_shape(conv1))
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    conv2 = ConvLSTM2D(64, (3, 3), activation='tanh', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(pool1)
    conv2 = ConvLSTM2D(64, (3, 3), activation='tanh', padding='same', return_sequences=True)(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    #
    conv3 = ConvLSTM2D(128, (3, 3), activation='tanh', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(pool2)
    conv3 = ConvLSTM2D(128, (3, 3), activation='tanh', padding='same', return_sequences=True)(conv3)

    up1 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv3), conv2], axis=-3)
    conv4 = ConvLSTM2D(64, (3, 3), activation='tanh', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(up1)
    conv4 = ConvLSTM2D(64, (3, 3), activation='tanh', padding='same', return_sequences=True)(conv4)
    #
    up2 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv4), conv1], axis=-3)
    conv5 = ConvLSTM2D(32, (3, 3), activation='tanh', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(up2)
    conv5 = ConvLSTM2D(32, (3, 3), activation='tanh', padding='same', return_sequences=True)(conv5)
    #unet_model = Model(inputs=inputs, outputs=conv5)
    ##
    l_prev = ConvLSTM2D(hidden_layers[0], (1, 1), activation='tanh', padding='same', return_sequences=True)(inputs)
    for n_hidden_neurons in hidden_layers[1:-1]:
        fc_new = ConvLSTM2D(n_hidden_neurons, (1, 1), activation='tanh', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(l_prev)
        # added by VP to decrease overfitting:
        #fc_new = Dropout(0.2)(fc_new)
        l_prev = fc_new
    fc_new = ConvLSTM2D(hidden_layers[-1], (1, 1), activation='tanh', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(l_prev)
    #fc_new = Dropout(0.2)(fc_new)
    l_prev = fc_new
    #mlp_model = Model(inputs=inputs, outputs=l_prev)
    print(K.int_shape(l_prev))
    print(K.int_shape(conv5))
    out_concat = concatenate([l_prev, conv5], axis=-3)
    print(K.int_shape(out_concat))
    # create common layers for softmax
    out2 = ConvLSTM2D(n_output, (1, 1), activation='tanh',padding='same', return_sequences=False)(out_concat) 
    # !!!! KEEP THIS RESHAPE AND PERMUTE FOR SOFTMAX !!!!
    out2 = core.Reshape((n_output, patch_height*patch_width))(out2)
    out2 = core.Permute((2,1))(out2)
    ############
    out2 = core.Activation('softmax')(out2)
    
    
    # get unified model
    model_out = Model(inputs, out2)
    
    if weights is not None:
        loss = weighted_categorical_crossentropy(weights)
        model_out.compile(optimizer=optimizer, loss=loss, metrics=[acc_labeled_only])
    else:
        model_out.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc_labeled_only])
    return model_out
    

    
def get_unet_mlp_convlstm(n_ch, n_timesteps, n_output, patch_height, patch_width, hidden_layers, optimizer="Adam", weights=None):
    '''
    Replacement of Conv2D (of get_unet_mlp_ts model) in Unet model and MLP model by ConvLSTM2D, and then concatenate only the last sequence of MLP and Unet
    '''
    # create u-net model
    inputs = Input((n_timesteps, n_ch, patch_height, patch_width))
    conv1 = ConvLSTM2D(32, (3, 3), activation='tanh', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs) # (None, 11, 32, 64, 64)
    #conv1bis = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs) ==> seems similar to ConvLSTM2D
    conv1 = ConvLSTM2D(32, (3, 3), activation='tanh', padding='same', return_sequences=True)(conv1) # (None, 11, 32, 64, 64)
    #print(K.int_shape(conv1))
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    conv2 = ConvLSTM2D(64, (3, 3), activation='tanh', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(pool1)
    conv2 = ConvLSTM2D(64, (3, 3), activation='tanh', padding='same', return_sequences=True)(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    #
    conv3 = ConvLSTM2D(128, (3, 3), activation='tanh', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(pool2)
    conv3 = ConvLSTM2D(128, (3, 3), activation='tanh', padding='same', return_sequences=True)(conv3)

    up1 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv3), conv2], axis=-3)
    conv4 = ConvLSTM2D(64, (3, 3), activation='tanh', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(up1)
    conv4 = ConvLSTM2D(64, (3, 3), activation='tanh', padding='same', return_sequences=True)(conv4)
    #
    up2 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv4), conv1], axis=-3)
    conv5 = ConvLSTM2D(32, (3, 3), activation='tanh', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(up2)
    conv5 = ConvLSTM2D(32, (3, 3), activation='tanh', padding='same', return_sequences=False)(conv5)
    #unet_model = Model(inputs=inputs, outputs=conv5)
    ##
    l_prev = ConvLSTM2D(hidden_layers[0], (1, 1), activation='tanh', padding='same', return_sequences=True)(inputs)
    for n_hidden_neurons in hidden_layers[1:-1]:
        fc_new = ConvLSTM2D(n_hidden_neurons, (1, 1), activation='tanh', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(l_prev)
        # added by VP to decrease overfitting:
        #fc_new = Dropout(0.2)(fc_new)
        l_prev = fc_new
    fc_new = ConvLSTM2D(hidden_layers[-1], (1, 1), activation='tanh', padding='same', return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(l_prev)
    #fc_new = Dropout(0.2)(fc_new)
    l_prev = fc_new
    #mlp_model = Model(inputs=inputs, outputs=l_prev)
    out_concat = concatenate([l_prev, conv5], axis=-3)
    # create common layers for softmax
    out2 = Conv2D(n_output, (1, 1), activation='relu',padding='same')(out_concat) # (None, 82, 64, 64)
    # !!!! KEEP THIS RESHAPE AND PERMUTE FOR SOFTMAX !!!!
    out2 = core.Reshape((n_output, patch_height*patch_width))(out2)
    out2 = core.Permute((2,1))(out2)
    ############
    out2 = core.Activation('softmax')(out2)
    
    
    # get unified model
    model_out = Model(inputs, out2)
    

    if weights is not None:
        loss = weighted_categorical_crossentropy(weights)
        model_out.compile(optimizer=optimizer, loss=loss, metrics=[acc_labeled_only])
    else:
        model_out.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc_labeled_only])
    return model_out
    
def get_unet_mlp_td_convlstm_withoutdropout(n_ch, n_timesteps, n_output, patch_height, patch_width, hidden_layers, optimizer="Adam", weights=None):
    '''
    Unet + MLP TimeDistributed, and then ConvLSTM2D. Error for dropouts if embedded in a TimeDistributed model
    '''
    # create u-net model
    inputs = Input((n_ch, patch_height, patch_width))
    #inputs = Input(shape=(n_ch, ))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    #conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    #conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    #conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    #conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    #
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    #conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    #unet_model = Model(inputs=inputs, outputs=conv5)
    #
    
    ###
    
    l_prev = Conv2D(hidden_layers[0], (1, 1), activation='relu', padding='same')(inputs)
    for n_hidden_neurons in hidden_layers[1:]:
        fc_new = Conv2D(n_hidden_neurons, (1, 1), activation='relu', padding='same')(l_prev)
        # added by VP to decrease overfitting:
        #fc_new = Dropout(0.2)(fc_new)
        l_prev = fc_new

    concat_unet_mlp = concatenate([l_prev, conv5], axis=1)
    #print(K.int_shape(concat_unet_mlp))
    model_mlp_unet = Model(inputs=inputs, outputs=concat_unet_mlp)
    # LSTM
    input_sequences = Input(shape=(n_timesteps, n_ch, patch_height, patch_width))
    
    #print(K.int_shape(input_sequences))
    processed_sequences = TimeDistributed(model_mlp_unet)(input_sequences)
    #print(K.int_shape(processed_sequences))
    out2 = ConvLSTM2D(n_output, (1, 1), activation='tanh', padding='same', return_sequences=False)(processed_sequences)
    out2 = core.Reshape((n_output, patch_height*patch_width))(out2)
    out2 = core.Permute((2,1))(out2)
    ############
    out2 = core.Activation('softmax')(out2)
    model_out = Model(inputs=input_sequences, outputs=out2)
    
    if weights is not None:
        loss = weighted_categorical_crossentropy(weights)
        model_out.compile(optimizer=optimizer, loss=loss, metrics=[acc_labeled_only])
    else:
        model_out.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc_labeled_only])
    return model_out
    
def get_unet_mlp_td_convlstm(n_ch, n_timesteps, n_output, patch_height, patch_width, hidden_layers, optimizer="Adam", weights=None):
    '''
    Unet + MLP TimeDistributed, and then ConvLSTM2D. To allow dropout, one TimeDistributed per layer
    '''
    input_sequences = Input(shape=(n_timesteps, n_ch, patch_height, patch_width))
    # create u-net model
    #inputs = Input((n_ch, patch_height, patch_width))
    #inputs = Input(shape=(n_ch, ))
    conv1 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(input_sequences)
    conv1 = TimeDistributed(Dropout(0.2))(conv1)
    conv1 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    #
    conv2 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(pool1)
    conv2 = TimeDistributed(Dropout(0.2))(conv2)
    conv2 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    #
    conv3 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(pool2)
    conv3 = TimeDistributed(Dropout(0.2))(conv3)
    conv3 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(conv3)

    up1 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv3), conv2], axis=2)
    conv4 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(up1)
    conv4 = TimeDistributed(Dropout(0.2))(conv4)
    conv4 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(conv4)
    #
    up2 = concatenate([TimeDistributed(UpSampling2D(size=(2, 2)))(conv4), conv1], axis=2)
    conv5 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(up2)
    conv5 = TimeDistributed(Dropout(0.2))(conv5)
    conv5 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(conv5)
    #unet_model = Model(inputs=inputs, outputs=conv5)
    #
    
    ###
    
    l_prev = TimeDistributed(Conv2D(hidden_layers[0], (1, 1), activation='relu', padding='same'))(input_sequences)
    for n_hidden_neurons in hidden_layers[1:]:
        fc_new = TimeDistributed(Conv2D(n_hidden_neurons, (1, 1), activation='relu', padding='same'))(l_prev)
        # added by VP to decrease overfitting:
        fc_new = TimeDistributed(Dropout(0.2))(fc_new)
        l_prev = fc_new
    #mlp_model = Model(inputs=inputs, outputs=l_prev)
    #print(K.int_shape(l_prev))
    #print(K.int_shape(conv5))
    
    concat_unet_mlp = concatenate([l_prev, conv5], axis=2)
    #print(K.int_shape(concat_unet_mlp))
    #model_mlp_unet = Model(inputs=inputs, outputs=concat_unet_mlp)
    # LSTM
    
    
    #print(K.int_shape(input_sequences))
    #processed_sequences = TimeDistributed(model_mlp_unet)(input_sequences)
    #print(K.int_shape(processed_sequences))
    out2 = ConvLSTM2D(n_output, (1, 1), activation='tanh', padding='same', return_sequences=False)(concat_unet_mlp)
    out2 = core.Reshape((n_output, patch_height*patch_width))(out2)
    out2 = core.Permute((2,1))(out2)
    ############
    out2 = core.Activation('softmax')(out2)
    model_out = Model(inputs=input_sequences, outputs=out2)
    
    if weights is not None:
        loss = weighted_categorical_crossentropy(weights)
        model_out.compile(optimizer=optimizer, loss=loss, metrics=[acc_labeled_only])
    else:
        model_out.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc_labeled_only])
    return model_out
    

def get_unet_mlp_lstm(n_ch, n_timesteps, n_output, patch_height, patch_width, hidden_layers, n_lstm, optimizer="Adam", weights = None):
    # create u-net model
    inputs = Input((n_ch, patch_height, patch_width))
    #inputs = Input(shape=(n_ch, ))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    #
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    #unet_model = Model(inputs=inputs, outputs=conv5)
    #
    
    ###
    
    l_prev = Conv2D(hidden_layers[0], (1, 1), activation='relu', padding='same')(inputs)
    for n_hidden_neurons in hidden_layers[1:]:
        fc_new = Conv2D(n_hidden_neurons, (1, 1), activation='relu', padding='same')(l_prev)
        # added by VP to decrease overfitting:
        fc_new = Dropout(0.2)(fc_new)
        l_prev = fc_new
    #mlp_model = Model(inputs=inputs, outputs=l_prev)
    print(K.int_shape(l_prev))
    print(K.int_shape(conv5))
    
    concat_unet_mlp = concatenate([l_prev, conv5], axis=1)
    print(K.int_shape(concat_unet_mlp))
    model_mlp_unet = Model(inputs=inputs, outputs=concat_unet_mlp)
    # LSTM
    input_sequences = Input(shape=(n_timesteps, n_ch, patch_height, patch_width)) # error in LSTM : Input 0 is incompatible with layer lstm_1: expected ndim=3, found ndim=5
    #input_sequences = Input(shape=(n_timesteps, n_ch)) # error in TimeDistributed : Dimension must be 2 but is 4 for 'time_distributed_1/conv2d_1/transpose
    print(K.int_shape(input_sequences))
    processed_sequences = TimeDistributed(model_mlp_unet)(input_sequences)
    print(K.int_shape(processed_sequences))
    processed_sequences_reshaped = core.Reshape((n_timesteps, (hidden_layers[-1]+32) * patch_height * patch_width))(processed_sequences)
    lstm1 = LSTM(n_lstm, return_sequences=True)(processed_sequences_reshaped) # LSTM expects input of shape (batch_size, timesteps, features) ==> 3 dims
    print(K.int_shape(lstm1)) # (None, 11, n_lstm) if return_sequences=True else (None, n_lstm)
    fc3 = Dense(n_output, activation='softmax')(lstm1)
    print(K.int_shape(fc3))
    model_out = Model(inputs=input_sequences, outputs=fc3)
    model_out.summary()

    if weights is not None:
        loss = weighted_categorical_crossentropy(weights)
        model_out.compile(optimizer=optimizer, loss=loss, metrics=[acc_labeled_only]) 
    else:
        model_out.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc_labeled_only])
    return model_out

    
#Define the neural network
def get_unet(n_ch,n_output,patch_height,patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    #
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    #
    conv6 = Conv2D(n_output, (1, 1), activation='relu',padding='same')(conv5)
    
    # !!!! KEEP THIS RESHAPE AND PERMUTE FOR SOFTMAX !!!!
    conv6 = core.Reshape((n_output,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=False)
    #model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def get_lin_net(n_ch,n_output,patch_height,patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    conv6 = Conv2D(n_output, (1, 1),padding='same')(inputs)
    conv6 = core.Reshape((n_output,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    #model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model


def get_sequential_unet_contours(n_timesteps, n_ch, n_output, patch_height, patch_width, class_weights=None):
    # create u-net model
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    #
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    #
    
    unet_model = Model(inputs=inputs, outputs=conv5)
    
    # share this model among temporal samples
    inputs_list = []
    out_list = []
    for i in range(n_timesteps):
        input_patch = Input((n_ch, patch_height, patch_width))
        inputs_list.append(input_patch)
        out_layer = unet_model(input_patch)
        out_list.append(out_layer)
    out = concatenate(out_list,axis=1)
    
    # create common layers for softmax
    out = Conv2D(n_output, (1, 1), activation='relu',padding='same')(out)
    # !!!! KEEP THIS RESHAPE AND PERMUTE FOR SOFTMAX !!!!
    out = core.Reshape((n_output,patch_height*patch_width))(out)
    out = core.Permute((2,1))(out)
    ############
    out = core.Activation('softmax')(out)
    
    # get unified model
    model_out = Model(inputs_list,out)
    
    ngpus = 1
    if ngpus > 1:
        model_out = make_parallel(model_out,ngpus)
    #model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    if class_weights is not None:
        loss = sobel_categorical_crossentropy(class_weights)
        model_out.compile(optimizer="Adam", loss=loss, metrics=['accuracy'])
    else:
        model_out.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    
    #model_out.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model_out


def get_sequential_mlp(n_ch, n_timesteps, n_output, patch_height, patch_width, hidden_layers, optimizer="Adam", weights=None):       
    # create model
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(hidden_layers[0], (1, 1), activation='relu', padding='same')(inputs)
    for i in range(1,len(hidden_layers)):
        conv1 = Dropout(0.2)(conv1)
        conv1 = Conv2D(hidden_layers[i], (1, 1), activation='relu', padding='same')(conv1)

    
    mlp_model = Model(inputs=inputs, outputs=conv1)
    
    # share this model among temporal samples
    inputs_list = []
    out_list = []
    for i in range(n_timesteps):
        input_patch = Input((n_ch, patch_height, patch_width))
        inputs_list.append(input_patch)
        out_layer = mlp_model(input_patch)
        out_list.append(out_layer)
    out = concatenate(out_list,axis=1)
    
    # create common layers for softmax
    out = Conv2D(n_output, (1, 1), activation='relu',padding='same')(out)
    # !!!! KEEP THIS RESHAPE AND PERMUTE FOR SOFTMAX !!!!
    out = core.Reshape((n_output,patch_height*patch_width))(out)
    out = core.Permute((2,1))(out)
    ############
    out = core.Activation('softmax')(out)
    
    # get unified model
    model_out = Model(inputs_list,out)
    
    if weights is not None:
        loss = weighted_categorical_crossentropy(weights)
        model_out.compile(optimizer=optimizer, loss=loss, metrics=[acc_labeled_only]) #+ [fscore_class(u_class) for u_class in range(n_output)])
    else:
        model_out.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc_labeled_only]) # + [fscore_class(u_class) for u_class in range(n_output)])
    return model_out
    
    #ngpus = 1
    #if ngpus > 1:
        #model_out = make_parallel(model_out,ngpus)    
    ##model_out.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    #if class_weights is not None:
        #loss = weighted_categorical_crossentropy(class_weights)
        #model_out.compile(optimizer="Adam", loss=loss, metrics=['accuracy'])
    #else:
        #model_out.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    ##model_out.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    #return model_out

def get_sequential_mlp_LSTM_error(n_ch, n_timesteps, n_output, hidden_layers, optimizer="Adam", weights=None):
    '''
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 33, 10)            0         
    _________________________________________________________________
    lstm_4 (LSTM)                (None, 33, 200)           168800    
    _________________________________________________________________
    lstm_5 (LSTM)                (None, 33, 100)           120400    
    _________________________________________________________________
    lstm_6 (LSTM)                (None, 50)                30200     
    _________________________________________________________________
    dense_13 (Dense)             (None, 17)                867       
    =================================================================
    Total params: 320,267
    
    '''
    # create model
    input_sequences = Input(shape=(n_timesteps, n_ch)) # # expected input_1 to have 3 dimensions, but got array with shape (4, 33, 10, 64, 64)
    lstm1 = LSTM(hidden_layers[0], dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(input_sequences)
    for i in range(1,len(hidden_layers)-1):
        lstm1 = LSTM(hidden_layers[i], dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(lstm1)
    lstm1 = LSTM(hidden_layers[-1], dropout=0.2, recurrent_dropout=0.2, return_sequences=False)(lstm1) # expected input_1 to have 3 dimensions, but got array with shape (4, 33, 10, 64, 64)
    # create common layers for softmax
    out = Dense(n_output, activation='softmax')(lstm1)
    # get unified model
    model_out = Model(input_sequences,out)
    if weights is not None:
        loss = weighted_categorical_crossentropy(weights)
        model_out.compile(optimizer=optimizer, loss=loss, metrics=[acc_labeled_only]) #+ [fscore_class(u_class) for u_class in range(n_output)])
    else:
        model_out.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc_labeled_only]) # + [fscore_class(u_class) for u_class in range(n_output)])
    return model_out

def get_sequential_mlp_LSTM(n_ch, n_timesteps, n_output, patch_height, patch_width, hidden_layers, optimizer="Adam", weights=None):
    '''
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 33, 10, 64, 64)    0         
    _________________________________________________________________
    conv_lst_m2d_1 (ConvLSTM2D)  (None, 33, 200, 64, 64)   168800    
    _________________________________________________________________
    conv_lst_m2d_2 (ConvLSTM2D)  (None, 33, 100, 64, 64)   120400    
    _________________________________________________________________
    conv_lst_m2d_3 (ConvLSTM2D)  (None, 33, 50, 64, 64)    30200     
    _________________________________________________________________
    conv_lst_m2d_4 (ConvLSTM2D)  (None, 17, 64, 64)        4624      
    _________________________________________________________________
    reshape_1 (Reshape)          (None, 17, 4096)          0         
    _________________________________________________________________
    permute_1 (Permute)          (None, 4096, 17)          0         
    _________________________________________________________________
    activation_1 (Activation)    (None, 4096, 17)          0         
    =================================================================
    Total params: 324,024
    Trainable params: 324,024
    
    '''
    # create model
    inputs = Input((n_timesteps, n_ch, patch_height, patch_width))
    l_prev = ConvLSTM2D(hidden_layers[0], (1, 1), activation='tanh', padding='same', return_sequences=True)(inputs)
    for n_hidden_neurons in hidden_layers[1:]:
        fc_new = ConvLSTM2D(n_hidden_neurons, (1, 1), activation='tanh', padding='same', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(l_prev)
        # added by VP to decrease overfitting:
        #fc_new = Dropout(0.2)(fc_new)
        l_prev = fc_new
    fc_new = ConvLSTM2D(n_output, (1, 1), activation='tanh', padding='same', return_sequences=False)(l_prev)
    
    # !!!! KEEP THIS RESHAPE AND PERMUTE FOR SOFTMAX !!!!
    out = core.Reshape((n_output,patch_height*patch_width))(fc_new)
    out = core.Permute((2,1))(out)
    ############
    out = core.Activation('softmax')(out)
    # get unified model
    model_out = Model(inputs, out)
    if weights is not None:
        loss = weighted_categorical_crossentropy(weights)
        model_out.compile(optimizer=optimizer, loss=loss, metrics=[acc_labeled_only])
    else:
        model_out.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc_labeled_only])
    return model_out
    

def get_sequential_unet_small(n_timesteps, n_ch, n_output, patch_height, patch_width, class_weights=None):
    # create u-net model
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    #
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    #
    
    unet_model = Model(inputs=inputs, outputs=conv5)
    
    # share this model among temporal samples
    inputs_list = []
    out_list = []
    for i in range(n_timesteps):
        input_patch = Input((n_ch, patch_height, patch_width))
        inputs_list.append(input_patch)
        out_layer = unet_model(input_patch)
        out_list.append(out_layer)
    out = concatenate(out_list,axis=1)
    
    # create common layers for softmax
    out = Conv2D(n_output, (1, 1), activation='relu',padding='same')(out)
    # !!!! KEEP THIS RESHAPE AND PERMUTE FOR SOFTMAX !!!!
    out = core.Reshape((n_output,patch_height*patch_width))(out)
    out = core.Permute((2,1))(out)
    ############
    out = core.Activation('softmax')(out)
    
    # get unified model
    model_out = Model(inputs_list,out)
    
    ngpus = 1
    if ngpus > 1:
        model_out = make_parallel(model_out,ngpus)    
    model_out.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    
    #model_out.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model_out



def get_sequential_unet(n_timesteps, n_ch, n_output, patch_height, patch_width, class_weights=None):
    # create u-net model
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Convolution2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv1)
    #
    conv2 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Convolution2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #
    conv4 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #
    conv4b = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool3)
    conv4b = Dropout(0.2)(conv4b)
    conv4b = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv4b)
    pool3b = MaxPooling2D(pool_size=(2, 2))(conv4b)
    #
    conv5 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool3b)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv5)
    #
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4b], axis=1)
    conv6b = Convolution2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv6b = Dropout(0.2)(conv6b)
    conv6b = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv6b)
    #
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6b), conv4], axis=1)
    conv6 = Convolution2D(64, (3, 3), activation='relu', padding='same')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv6)
    #
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv7)
    #
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Convolution2D(16, (3, 3), activation='relu', padding='same')(up4)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv8)
    
    unet_model = Model(inputs=inputs, outputs=conv8)
    
    # share this model among temporal samples
    inputs_list = []
    out_list = []
    for i in range(n_timesteps):
        input_patch = Input((n_ch, patch_height, patch_width))
        inputs_list.append(input_patch)
        out_layer = unet_model(input_patch)
        out_list.append(out_layer)
    out = concatenate(out_list,axis=1)
    
    # create common layers for softmax
    out = Conv2D(n_output, (1, 1), activation='relu',padding='same')(out)
    # !!!! KEEP THIS RESHAPE AND PERMUTE FOR SOFTMAX !!!!
    out = core.Reshape((n_output,patch_height*patch_width))(out)
    out = core.Permute((2,1))(out)
    ############
    out = core.Activation('softmax')(out)
    
    # get unified model
    model_out = Model(inputs_list,out)
    
    ngpus = 1
    if ngpus > 1:
        model_out = make_parallel(model_out,ngpus)
    #model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    if class_weights is not None:
        loss = weighted_categorical_crossentropy(class_weights)
        model_out.compile(optimizer="Adam", loss=loss, metrics=['accuracy'])
    else:
        model_out.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    
    #model_out.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model_out

def get_pixelwise_fc_lstm(hidden_layer_sizes, n_lstm, n_bands, n_timestamps, n_output, class_weights=None):

    inputs = Input(shape=(n_bands,))
    l_prev = inputs
    for nh in hidden_layer_sizes:
        fc_new = Dense(nh, activation='relu')(l_prev)
        l_prev = fc_new
    #fc2 = Dense(n_hidden, activation='relu')(fc1)
    model = Model(inputs=inputs, outputs=l_prev)

    input_sequences = Input(shape=(n_timestamps, n_bands))
    processed_sequences = TimeDistributed(model)(input_sequences)
    lstm1 = LSTM(n_lstm)(processed_sequences)
    fc3 = Dense(n_output, activation='softmax')(lstm1)

    model_out = Model(inputs=input_sequences, outputs=fc3)
    if class_weights is not None:
        loss = weighted_categorical_crossentropy(class_weights)
        model_out.compile(optimizer="Adam", loss=loss, metrics=['accuracy'])
    else:
        model_out.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model_out


def get_pixelwise_fc_timeconcat(hidden_layer_sizes, n_bands, n_timestamps, n_output, class_weights=None):

    inputs = Input(shape=(n_bands,))
    l_prev = inputs
    for nh in hidden_layer_sizes:
        fc_new = Dense(nh, activation='relu')(l_prev)
        l_prev = fc_new
    #fc2 = Dense(n_hidden, activation='relu')(fc1)
    model = Model(inputs=inputs, outputs=l_prev)

    input_sequences = Input(shape=(n_timestamps, n_bands))
    processed_sequences = TimeDistributed(model)(input_sequences)
    rs1 = Flatten()(processed_sequences)
    fc3 = Dense(n_output, activation='softmax')(rs1)

    model_out = Model(inputs=input_sequences, outputs=fc3)
    if class_weights is not None:
        loss = weighted_categorical_crossentropy(class_weights)
        model_out.compile(optimizer="Adam", loss=loss, metrics=['accuracy'])
    else:
        model_out.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model_out


def get_pixelwise_fc_time_nosharedweights(hidden_layer_sizes, n_bands, n_timestamps, n_output, class_weights=None):

    input_sequences = Input(shape=(n_timestamps, n_bands))
    l_prev = Flatten()(input_sequences)
    for nh in hidden_layer_sizes:
        fc_new = Dense(nh, activation='relu')(l_prev)
        l_prev = fc_new
    fc3 = Dense(n_output, activation='softmax')(l_prev)

    model_out = Model(inputs=input_sequences, outputs=fc3)
    if class_weights is not None:
        loss = weighted_categorical_crossentropy(class_weights)
        model_out.compile(optimizer="Adam", loss=loss, metrics=['accuracy'])
    else:
        model_out.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model_out


