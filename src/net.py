from VggFace_model import vgg_face

def make_net(net_path, split, mean, opt):
    '''
    net_path: path for the prototxt file of the net
    split: 'train' / 'val' / 'test'
    mean: channel mean for the datasets used   
    '''
    print "Writing prototxt file for train net..."
    with open(net_path, 'w') as f:
        if opt.model_name == 'VggFace':
            f.write( str ( vgg_face(split, mean, opt) ) )
        else:
            raise ValueError('Unrecognized network type')