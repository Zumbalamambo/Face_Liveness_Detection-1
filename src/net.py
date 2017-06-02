import caffe
from caffe import layers as L, params as P

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, lr_shrink=10):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad,
                         param=[dict(lr_mult=1/lr_shrink, decay_mult=1/lr_shrink), 
                                dict(lr_mult=2/lr_shrink, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def fc_relu_dropout(bottom, nout, ratio, lr_shrink=10):
    fc = L.InnerProduct(bottom, num_output=nout, 
                        param=[dict(lr_mult=1/lr_shrink, decay_mult=1/lr_shrink), 
                               dict(lr_mult=2/lr_shrink, decay_mult=0)])
    relu = L.ReLU(fc, in_place=True)
    dropout = L.Dropout(relu, dropout_ratio=ratio, in_place=True)
    return fc, relu, dropout

def vgg_face(split, batch_size):
    n = caffe.NetSpec()

    # config python data layer (mean values are about the same for every subset)
    pydata_params = dict(split=split, mean=(89.7647, 106.3760, 139.7756),
                         data_root='../data/')

    n.data, n.label = L.Python(module='faceData_layers', layer='FaceDataLayer', 
                               ntop=2, param_str=str(pydata_params))

    # vgg-face net
    # conv layers
    lr_shrink_ratio = 10
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, lr_shrink=lr_shrink_ratio)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, lr_shrink=lr_shrink_ratio)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, lr_shrink=lr_shrink_ratio)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, lr_shrink=lr_shrink_ratio)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, lr_shrink=lr_shrink_ratio)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, lr_shrink=lr_shrink_ratio)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, lr_shrink=lr_shrink_ratio)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, lr_shrink=lr_shrink_ratio)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, lr_shrink=lr_shrink_ratio)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, lr_shrink=lr_shrink_ratio)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, lr_shrink=lr_shrink_ratio)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, lr_shrink=lr_shrink_ratio)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, lr_shrink=lr_shrink_ratio)
    n.pool5 = max_pool(n.relu5_3)

    # drop out and fc layers
    n.fc6, n.relu6, n.drop6 = fc_relu_dropout(n.pool5, 4096, 0.5, lr_shrink=lr_shrink_ratio)
    n.fc7, n.relu7, n.drop7 = fc_relu_dropout(n.fc6, 4096, 0.5, lr_shrink=lr_shrink_ratio)
    n.fc8_face = L.InnerProduct(n.fc7, num_output=1024, 
                                param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], 
                                weight_filler=dict(type='gaussian', std=0.01), 
                                bias_filler=dict(type='constant', value=0)
                                )
    n.fc9_face = L.InnerProduct(n.fc8_face, num_output=2, 
                                param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], 
                                weight_filler=dict(type='gaussian', std=0.01), 
                                bias_filler=dict(type='constant', value=0)
                                )

    # loss layer
    n.loss = L.SoftmaxWithLoss(n.fc9_face, n.label)

    return n.to_proto()

def make_net(train_net_path, val_net_path):
    with open(train_net_path, 'w') as f:
        f.write( str ( vgg_face('train', 5) ) )

    with open(val_net_path, 'w') as f:
        f.write( str( vgg_face('val', 50) ) )