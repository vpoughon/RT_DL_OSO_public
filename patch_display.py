import keras
from PIL import Image
import numpy as np
import random

def make_mosaic_result(img,mask,preds):
    assert (img.shape[0]== mask.shape[0] and img.shape[0]== preds.shape[0])
    assert (img.shape[1]== mask.shape[1] and img.shape[1]== preds.shape[1])
    assert (img.shape[2]== mask.shape[2] and img.shape[2]== preds.shape[2])
    nb_samples = img.shape[0]
    nb_rows = min(5,nb_samples)
    nb_cols = 3*(np.floor(nb_samples/nb_rows)).astype(np.uint8)
    patch_x = img.shape[2]
    patch_y = img.shape[1]
    mosaic = np.zeros((nb_rows*patch_y,nb_cols*patch_x,3), dtype=np.uint8) 
    for i in range(nb_samples):
        r = i % nb_rows
        c = (int)((i - (i %  nb_rows))/nb_rows)
        if c*3 >= nb_cols:
            continue
        mosaic[patch_y*r:patch_y*(r+1),patch_x*3*c:patch_x*(3*c+1),:] = img[i,...]
        mosaic[patch_y*r:patch_y*(r+1),patch_x*(3*c+1):patch_x*(3*c+2),:] = mask[i,...]
        mosaic[patch_y*r:patch_y*(r+1),patch_x*(3*c+2):patch_x*(3*c+3),:] = preds[i,...]
    return mosaic   
        
    

# reformat and save image
def save_img(data,filename):
    img = None
    if(len(data.shape)==3): #height*width*channels
        img = None
        if data.shape[2]==1:  #in case it is black and white
            data = np.reshape(data,(data.shape[0],data.shape[1]))
        if np.max(data)>1:
            img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
        else:
            img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    else:
        img = Image.fromarray(data.astype(np.uint8)) 
    img.save(filename + '.png')
    return img

def format_patches_for_display_colormap(patch, gt_patch, pred_patch, input_ch=[0,1,2], input_gain=1, colormap={}, thresh_output=False):
    assert(patch.shape[0] == gt_patch.shape[0] and patch.shape[0] == pred_patch.shape[0])
    assert(np.array_equal(gt_patch.shape,pred_patch.shape))
    assert(len(input_ch)>0 and np.max(input_ch)<patch.shape[1])

    # make sure channels are well formatted
    if len(input_ch) == 1:
        input_ch = np.repeat(input_ch,3)    
    if len(input_ch) > 3:
        input_ch = input_ch[0:2]
    #if len(colormap) < gt_patch.shape[2]:
    #    for i in range(len(colormap),gt_patch.shape[2]):
    #        colormap[i] = [random.randint(0,255),
    #                       random.randint(0,255),
    #                       random.randint(0,255)]     

    # initialize output images
    disp_patch = np.zeros((patch.shape[0], patch.shape[2], patch.shape[3],3))
    disp_gt_patch = np.zeros((patch.shape[0], patch.shape[2], patch.shape[3],3))
    disp_pred_patch = np.zeros((patch.shape[0], patch.shape[2], patch.shape[3],3))
           
    # format patch
    patch = np.transpose(patch,(0,2,3,1))
    for i in range(len(input_ch)):
        disp_patch[...,i] = patch[...,input_ch[i]]*255*input_gain
    disp_patch = disp_patch.astype('uint8')
    # format gt_patch
    # TODO: case where patch is already squared
    gt_patch = np.reshape(gt_patch, [gt_patch.shape[0],
                                     disp_gt_patch.shape[1], 
                                     disp_gt_patch.shape[2],
                                     gt_patch.shape[2]])
    #print(len(colormap))
    for i in range(len(colormap)):
        for ch in range(3):
            disp_gt_patch[...,ch] += gt_patch[...,i]*colormap[i][ch]
    disp_gt_patch = disp_gt_patch.astype('uint8')
    # format pred_patch
    # TODO: case where patch is already squared
    pred_patch = np.reshape(pred_patch, [pred_patch.shape[0],
                                     disp_pred_patch.shape[1], 
                                     disp_pred_patch.shape[2],
                                     pred_patch.shape[2]])
    # hard thresh required
    pred_patch_argmax = np.argmax(pred_patch[...,:len(colormap)],axis=3)
    pred_patch.fill(0)

    for i in range(len(colormap)):
        pred_patch[...,i] = (pred_patch_argmax==i).astype('float32')
        for ch in range(3):
            disp_pred_patch[...,ch] += pred_patch[...,i]*colormap[i][ch]
    disp_pred_patch = disp_pred_patch.astype('uint8')

#    print("RGB: {},{}, GT: {},{}, PRED:{},{}".format(np.min(disp_patch),
#                                                     np.max(disp_patch),
#                                                     np.min(disp_gt_patch),
#                                                     np.max(disp_gt_patch),
#                                                     np.min(disp_pred_patch),
#                                                     np.max(disp_pred_patch)
#                                                     ))
    return disp_patch, disp_gt_patch, disp_pred_patch

def format_patches_for_display(patch, gt_patch, pred_patch, input_ch=[0,1,2], output_ch=[0,1,2], input_gain=1, thresh_output=False):
    assert(patch.shape[0] == gt_patch.shape[0] and patch.shape[0] == pred_patch.shape[0])
    assert(np.array_equal(gt_patch.shape,pred_patch.shape))
    assert(len(input_ch)>0 and np.max(input_ch)<patch.shape[1])
    assert(len(output_ch)>0 and np.max(output_ch)<gt_patch.shape[1])
    # make sure channels are well formatted
    if len(input_ch) == 1:
        input_ch = np.repeat(input_ch,3)
    if len(output_ch) == 1:
        output_ch = np.repeat(output_ch,3)       
    if len(input_ch) > 3:
        input_ch = input_ch[0:2]
    if len(output_ch) > 3:
        output_ch = output_ch[0:2]
    # initialize output images
    disp_patch = np.zeros((patch.shape[0], patch.shape[2], patch.shape[3],3))
    disp_gt_patch = np.zeros((patch.shape[0], patch.shape[2], patch.shape[3],3))
    disp_pred_patch = np.zeros((patch.shape[0], patch.shape[2], patch.shape[3],3))
           
    # format patch
    patch = np.transpose(patch,(0,2,3,1))
    for i in range(len(input_ch)):
        disp_patch[...,i] = patch[...,input_ch[i]]*255*input_gain
    disp_patch = disp_patch.astype('uint8')
    # format gt_patch
    # TODO: case where patch is already squared
    gt_patch = np.reshape(gt_patch, [gt_patch.shape[0],
                                     disp_gt_patch.shape[1], 
                                     disp_gt_patch.shape[2],
                                     gt_patch.shape[2]])
    for i in range(len(output_ch)):
        disp_gt_patch[...,i] = gt_patch[...,output_ch[i]]*255
    disp_gt_patch = disp_gt_patch.astype('uint8')
    # format pred_patch
    # TODO: case where patch is already squared
    pred_patch = np.reshape(pred_patch, [pred_patch.shape[0],
                                     disp_pred_patch.shape[1], 
                                     disp_pred_patch.shape[2],
                                     pred_patch.shape[2]])
    for i in range(len(output_ch)):
        disp_pred_patch[...,i] = pred_patch[...,output_ch[i]]
    if thresh_output:
        disp_pred_patch = (disp_pred_patch > 0.5).astype('uint8')
    disp_pred_patch *= 255
    disp_pred_patch = disp_pred_patch.astype('uint8')
    
    return disp_patch, disp_gt_patch, disp_pred_patch