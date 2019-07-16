import tensorflow as tf
import cv2

import numpy as np
import os
import random
import sys
import nn

from sklearn.model_selection import train_test_split

faces_my_path = './faces_my'
faces_other_path = './faces_other'
batch_size = 128          # 每次取128张图片
learning_rate = 0.01        # 学习率
size = 64                 # 图片大小64*64*3
imgs1 = []                 # 存放人脸图片
labs1 = []                 # 存放人脸图片对应的标签
imgs3 = []
labs3 = []
x = tf.placeholder(tf.float32, [None, size, size, 3])  # 输入X：64*64*3
y_ = tf.placeholder(tf.float32, [None, 2])  # 输出Y_：1*2
keep_prob_fifty = tf.placeholder(tf.float32)  # 50%，即0.5
keep_prob_seventy_five = tf.placeholder(tf.float32)  # 75%，即0.75


def bbreg(boundingbox, reg):
    """Calibrate bounding boxes"""
    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return boundingbox


def generateBoundingBox(imap, reg, scale, t):
    """Use heatmap to generate bounding boxes"""
    stride = 2
    cellsize = 12

    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:, :, 0])
    dy1 = np.transpose(reg[:, :, 1])
    dx2 = np.transpose(reg[:, :, 2])
    dy2 = np.transpose(reg[:, :, 3])
    y, x = np.where(imap >= t)
    if y.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    score = imap[(y, x)]
    reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))
    if reg.size == 0:
        reg = np.empty((0, 3))
    bb = np.transpose(np.vstack([y, x]))
    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize - 1 + 1) / scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])
    return boundingbox, reg


# function pick = nms(boxes,threshold,type)
def nms(boxes, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick


# function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
def pad(total_boxes, w, h):
    """Compute the padding coordinates (pad the bounding boxes to square)"""
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(ex > w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w

    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h

    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1

    tmp = np.where(y < 1)
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
    y[tmp] = 1

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph


# function [bboxA] = rerec(bboxA)
def rerec(bboxA):
    """Convert bboxA to square."""
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    l = np.maximum(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
    return bboxA


def create_mtcnn(sess, model_path):
    with tf.variable_scope('pnet'):
        data = tf.placeholder(tf.float32, (None, None, None, 3), 'input')
        pnet = nn.PNet({'data': data})
        pnet.load(os.path.join(model_path, 'det1.npy'), sess)
    with tf.variable_scope('rnet'):
        data = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
        rnet = nn.RNet({'data': data})
        rnet.load(os.path.join(model_path, 'det2.npy'), sess)
    with tf.variable_scope('onet'):
        data = tf.placeholder(tf.float32, (None, 48, 48, 3), 'input')
        onet = nn.ONet({'data': data})
        onet.load(os.path.join(model_path, 'det3.npy'), sess)

    pnet_fun = lambda img: sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0': img})
    rnet_fun = lambda img: sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0': img})
    onet_fun = lambda img: sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'),
                                    feed_dict={'onet/input:0': img})

    return pnet_fun, rnet_fun, onet_fun


def detect_face(img, minsize, pnet, rnet, onet, threshold, factor):
    """Detects faces in an image, and returns bounding boxes and points for them.
    img: input image
    minsize: minimum faces' size
    pnet, rnet, onet: caffemodel
    threshold: threshold=[th1, th2, th3], th1-3 are three steps's threshold
    factor: the factor used to create a scaling pyramid of face sizes to detect in the image.
    """
    factor_count = 0
    total_boxes = np.empty((0, 9))
    h = img.shape[0]
    w = img.shape[1]
    minl = np.amin([h, w])
    m = 12.0 / minsize
    minl = minl * m
    # create scale pyramid
    scales = []
    while minl >= 12:
        scales += [m * np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1

    # first stage
    for scale in scales:
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        im_data = cv2.resize(img, (hs, ws), interpolation=cv2.INTER_AREA)
        im_data = (im_data - 127.5) * 0.0078125
        img_x = np.expand_dims(im_data, 0)
        img_y = np.transpose(img_x, (0, 2, 1, 3))
        out = pnet(img_y)
        out0 = np.transpose(out[0], (0, 2, 1, 3))
        out1 = np.transpose(out[1], (0, 2, 1, 3))

        boxes, _ = generateBoundingBox(out1[0, :, :, 1].copy(), out0[0, :, :, :].copy(), scale, threshold[0])

        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage
        tempimg = np.zeros((24, 24, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_AREA)
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
        out = rnet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out1[1, :]
        ipass = np.where(score > threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
        mv = out0[:, ipass[0]]
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
            total_boxes = rerec(total_boxes.copy())

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # third stage
        total_boxes = np.fix(total_boxes).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
        tempimg = np.zeros((48, 48, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
        out = onet(tempimg1)
        out0 = np.transpose(out[0])
        out2 = np.transpose(out[2])
        score = out2[1, :]
        ipass = np.where(score > threshold[2])
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
        mv = out0[:, ipass[0]]

        if total_boxes.shape[0] > 0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
            pick = nms(total_boxes.copy(), threshold[2], 'Min')
            total_boxes = total_boxes[pick, :]

    return total_boxes
    # This method is kept for debugging purpose


#     h=img.shape[0]
#     w=img.shape[1]
#     hs, ws = sz
#     dx = float(w) / ws
#     dy = float(h) / hs
#     im_data = np.zeros((hs,ws,3))
#     for a1 in range(0,hs):
#         for a2 in range(0,ws):
#             for a3 in range(0,3):
#                 im_data[a1,a2,a3] = img[int(floor(a1*dy)),int(floor(a2*dx)),a3]
#     return im_data

minsize = 20  # minimum size of face
thresh = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor face image pyramid 图像缩小尺度
margin = 44

def readData(path , imgs,labs, h = size , w = size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            top,bottom,left,right = getPaddingSize(img)
            """放大图片扩充图片边缘部分"""
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))
            imgs.append(img)                 # 一张张人脸图片加入imgs列表中
            labs.append(path)                # 一张张人脸图片对应的path，即文件夹名faces_my和faces_other，即标签

def getPaddingSize(img):
    height, width, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(height, width)

    if width < longest:
        tmp = longest - width
        left = tmp // 2
        right = tmp - left
    elif height < longest:
        tmp = longest - height
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def cnnLayer():
    W1 = tf.Variable(tf.random_normal([3, 3, 3, 32]))                 # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = tf.Variable(tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')+b1)    # 64*64*32，卷积提取特征，增加通道数
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 32*32*32，池化降维，减小复杂度
    drop1 = tf.nn.dropout(pool1, keep_prob_fifty)

    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))  # 卷积核大小(3,3)， 输入通道(32)， 输出通道(64)
    b2 = tf.Variable(tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.conv2d(drop1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)        # 32*32*64
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')       # 16*16*64
    drop2 = tf.nn.dropout(pool2, keep_prob_fifty)

    W3 = tf.Variable(tf.random_normal([3, 3, 64, 64]))  # 卷积核大小(3,3)， 输入通道(64)， 输出通道(64)
    b3 = tf.Variable(tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.conv2d(drop2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3)        # 16*16*64
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')       # 8*8*64=4096
    drop3 = tf.nn.dropout(pool3, keep_prob_fifty)

    Wf = tf.Variable(tf.random_normal([8*8*64,512]))     # 输入通道(4096)， 输出通道(512)
    bf = tf.Variable(tf.random_normal([512]))
    drop3_flat = tf.reshape(drop3, [-1, 8*8*64])         # -1表示行随着列的需求改变，1*4096
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)   # [1,4096]*[4096,512]=[1,512]
    dropf = tf.nn.dropout(dense, keep_prob_seventy_five)

    Wout = tf.Variable(tf.random_normal([512,2]))        # 输入通道(512)， 输出通道(2)
    bout = tf.Variable(tf.random_normal([2]))
    out = tf.add(tf.matmul(dropf, Wout), bout)     # (1,512)*(512,2)=(1,2) ,跟y_ [0,1]、[1,0]比较给出损失
    return out

def face_recognise(image):
    res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_fifty: 1.0, keep_prob_seventy_five: 1.0})

    if res[0] == 1:
        return "yes"
    else:
        return "no"




if __name__ == '__main__':

    readData(faces_my_path,imgs1,labs1)
    readData(faces_other_path,imgs3,labs3)
    imgs1 = np.array(imgs1)  # 将图片数据与标签转换成数组
    labs1 = np.array([[0, 1] if lab == faces_my_path else [1, 0] for lab in labs1])  # 标签：[0,1]表示是我的人脸，[1,0]表示其他的人脸

    train_x_1, test_x_1, train_y, test_y = train_test_split(imgs1, labs1, test_size=0.05,
                                                            random_state=random.randint(0, 100))
    train_x_2 = train_x_1.reshape(train_x_1.shape[0], size, size, 3)  # 参数：图片数据的总数，图片的高、宽、通道
    test_x_2 = test_x_1.reshape(test_x_1.shape[0], size, size, 3)
    train_x = train_x_2.astype('float32') / 255.0                      # 归一化
    test_x = test_x_2.astype('float32') / 255.0


    print('训练集大小:%s, 测试集大小:%s' % (len(train_x), len(test_x)))

    num_batch = len(train_x_1) // batch_size
    out = cnnLayer()
    predict = tf.argmax(out, 1)

    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        mtcnn_model_path = 'mtcnn_model/'
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            # 目的是将这个图设置为默认图，会话设置成默认对话，这样的话在with语句的外面也能使用这个会话执行。
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            # 记录设备指派情况: tf.ConfigProto(log_device_placement=True)
            # 设置tf.ConfigProto()
            # 中参数log_device_placement = True, 可以获取到operations和Tensor被指派到哪个设备(几号CPU或几号GPU)
            # 上运行, 会在终端打印出各项操作是在哪个设备上运行的。
            with sess.as_default():
                pnet, rnet, onet = create_mtcnn(sess, mtcnn_model_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (600, 600))

        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes = detect_face(img, minsize, pnet, rnet, onet, thresh, factor)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if len(bounding_boxes) > 0:
            for face in range(len(bounding_boxes)):
                det = np.squeeze(bounding_boxes[face, 0:4])
                (startX, startY, endX, endY) = det.astype("int")
                y = startY - 10 if startY - 10 > 10 else startY + 10
                if not len(det):

                    key = cv2.waitKey(30)
                    if key == 27:
                        sys.exit(0)

                x1 = startX
                y1 = startY
                x2 = endX
                y2 = endY
                face = img[x1:x2, y1:y2]
                face = cv2.resize(face, (size, size))
                saver = tf.train.Saver()
                sess = tf.Session()
                saver.restore(sess, tf.train.latest_checkpoint('./ckp'))

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                print(face_recognise(face))
                if face_recognise(face) == "yes":
                    cv2.putText(img, 'nijiarong', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(img, 'other', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('image', img)
                key = cv2.waitKey(30)
                if key == 27:
                    sys.exit(0)


    sess.close()
