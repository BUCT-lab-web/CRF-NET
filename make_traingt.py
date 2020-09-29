def get_n_hls_colors(num):
    import random
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        L = 50 + random.random() * 10
        _hls = [h / 360.0, L / 100.0, s / 100.0]
        hls_colors.append(_hls)
        i += step
    return hls_colors


def ncolors(num):
    import colorsys
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return np.array(rgb_colors)

def getpixels(xsite, ysite, xlen, ylen):
    xpixels = np.arange(xsite, xsite+xlen, 1)
    ypixels = np.arange(ysite, ysite+ylen, 1)
    return xpixels, ypixels

def ifOk1(block, classes):
    p = np.where(block==classes)[0]
    if len(p) == (block.shape[0]*block.shape[1]) :
        return True
    else:
        return False
    
def ifOK2(i, j, position):
    if position == []:
        return True
    if position != []:
        for k in range(len(position)):
#            print(position[k][0], position[k][1])
            f =(i-position[k][0])**2 + (j-position[k][1])**2
            if f <= 121:
                return False
        return True
    
def getblock(label, classes):
    r, c = label.shape
    position = []
    zero = np.zeros((r, c))
    x = np.array([3, 5, 9, 15])
    y = np.array([5, 3, 9, 15])
    pos = np.where(label == classes)
    train_number = int(len(pos[0])*0.1)
    count = 0
    while count < train_number:
        site = np.random.randint(0, len(pos[0]))
        xsite, ysite = pos[0][site], pos[1][site]
        xlen = x[np.random.randint(0, len(x))]
        ylen = y[np.random.randint(0, len(y))]
        if xsite+xlen < r and ysite+ylen < c: 
            xpos, ypos = getpixels(xsite, ysite, xlen, ylen)
            block = label[np.ix_(xpos, ypos)]
#        print(block.shape, len(xpos), len(ypos))
            if ifOk1(block, classes) and ifOK2(xsite, ysite, position):
                zero[np.ix_(xpos, ypos)] = classes
                count += block.shape[0]*block.shape[1]
                position.append([xsite, ysite])
    if count < train_number:
        num = train_number - count
        return zero
    else:
        return zero
    
def gt_to_img(gt, path, COLOR_DICT):
    import cv2 as cv
    W, H = gt.shape
    seg_img = np.ones((W, H, 3), dtype=np.uint8)*240
    for index, item in enumerate(COLOR_DICT):
        x, y = np.where(gt == index+1)
        seg_img[x, y, 0] = item[0]
        seg_img[x, y, 1] = item[1]
        seg_img[x, y, 2] = item[2]
    seg_img = seg_img[:, :, ::-1]
    cv.imwrite(path, seg_img)
    
def select(label):
    r, c = label.shape
    train = np.zeros((r, c))
    gt = label.reshape(r*c)
    classes = np.unique(gt) - 1
    for i in range(len(classes)):
        pos = np.where(label == i+1)
        im = np.zeros((r, c))
        im[pos] = i+1
        train += getblock(im, i+1)
    return train

if __name__ == '__main__':
    import scipy.io as sio
    import numpy as np
    d =sio.loadmat('PaviaU_gt.mat')

    label = d['paviaU_gt']
    
    im = select(label)
    sio.savemat('GetBlock_gt.mat', {'GetBlock_gt': im})
    #color = ncolors(9)
    #gt_to_img(im, 'im.png', color)

        
            

    



























