import anchor_target_layer
import numpy as np
import read_pascal_voc

rpn=[]
feat_stride        = 16
anchor_scale       = [ 8, 16, 32 ]
imageNameFile = "../../../Datasets/VOCdevkit/VOC2012/ImageSets/Main/aeroplane_train.txt"
vocPath       = "../../../Datasets/VOCdevkit/VOC2012"

Image_data,boundingBX_labels,im_dims=read_pascal_voc.prepareBatch(0,2,imageNameFile,vocPath)
print(Image_data,boundingBX_labels,im_dims)

_label,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights=anchor_target_layer.anchor_target_layer_python(rpn_cls_score=Image_data,gt_boxes=boundingBX_labels,im_dims=im_dims,feat_strides=feat_stride,anchor_scales=anchor_scale)
# (rpn_label)
