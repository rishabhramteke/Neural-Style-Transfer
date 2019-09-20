#author : Rishabh Ramteke 

#importing libraries
import tensorflow as tf
import numpy as np
import cv2

#hyperparameters
alpha = 0.000004 #initial : 0.000004 
beta = 0.0001

#functions
def gram_matrix(l,i,j): # l is layer; vectorized feature map i and j;
    g = tf.reshape(l, (i, j))
    return tf.matmul(tf.transpose(g), g)

def contribution_of_layer_to_loss(M,N, G, A):
    contribution = (1.0 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
    return contribution

#Function to define Bias variable is required, otherwise getting some random error
def bias_variable(shape):
    b_initial=tf.constant(0.1,shape=shape)
    return tf.Variable(b_initial)

#Function to perform convolution
def conv2d(input,weight):
    return tf.nn.conv2d(input,weight,strides=[1, 1, 1, 1], padding='SAME')

#Pooling Function
def max_pool(input):
    return tf.nn.max_pool(input,strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1] , padding='SAME')

#function for convolution layer
def conv_layer(input,shape):
    W=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    b=bias_variable([shape[3]])
    return tf.nn.relu(tf.nn.conv2d(x,W,strides=[1, 1, 1, 1], padding='SAME')+b)



class vgg16:

    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs_update-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            shape_1=[3, 3, 3, 64]
            W = tf.Variable(tf.truncated_normal(shape_1,stddev=0.1))
            b = bias_variable([shape_1[3]])
            conv = tf.nn.relu(tf.nn.conv2d(images, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv1_1 = conv
            self.parameters += [W, b]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            shape_2=[3, 3, 64, 64]
            W = tf.Variable(tf.truncated_normal(shape_2,stddev=0.1))
            b = bias_variable([shape_2[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.conv1_1, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv1_2 = conv
            self.parameters += [W, b]

        # pool1
        self.pool1 = max_pool(self.conv1_2)

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            shape_3=[3, 3, 64, 128]
            W = tf.Variable(tf.truncated_normal(shape_3,stddev=0.1))
            b = bias_variable([shape_3[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.pool1, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv2_1 = conv
            self.parameters += [W, b]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            shape_4=[3, 3, 128, 128]
            W = tf.Variable(tf.truncated_normal(shape_4,stddev=0.1))
            b = bias_variable([shape_4[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.conv2_1, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv2_2 =conv
            self.parameters += [W, b]

        # pool2
        self.pool2 = max_pool(self.conv2_2)
        
        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            shape_5=[3, 3, 128, 256]
            W = tf.Variable(tf.truncated_normal(shape_5,stddev=0.1))
            b = bias_variable([shape_5[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.pool2, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv3_1 = conv
            self.parameters += [W, b]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            shape_6=[3, 3, 256, 256]
            W = tf.Variable(tf.truncated_normal(shape_6,stddev=0.1))
            b = bias_variable([shape_6[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.conv3_1, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv3_2 = conv
            self.parameters += [W, b]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            shape_7=[3, 3, 256, 256]
            W = tf.Variable(tf.truncated_normal(shape_7,stddev=0.1))
            b = bias_variable([shape_7[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.conv3_2, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv3_3 =conv
            self.parameters += [W, b]

        # pool3
        self.pool3 = max_pool(self.conv3_3)
        
        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            shape_8=[3, 3, 256, 512]
            W = tf.Variable(tf.truncated_normal(shape_8,stddev=0.1))
            b = bias_variable([shape_8[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.pool3, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv4_1 = conv
            self.parameters += [W, b]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            shape_9=[3, 3, 512, 512]
            W = tf.Variable(tf.truncated_normal(shape_9,stddev=0.1))
            b = bias_variable([shape_9[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.conv4_1, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv4_2 = conv
            self.parameters += [W, b]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            shape_10=[3, 3, 512, 512]
            W = tf.Variable(tf.truncated_normal(shape_10,stddev=0.1))
            b = bias_variable([shape_10[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.conv4_2 , W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv4_3 =conv
            self.parameters += [W, b]

        # pool4
        self.pool4 = max_pool(self.conv4_3)
        
        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            shape_11=[3, 3, 512, 512]
            W = tf.Variable(tf.truncated_normal(shape_11,stddev=0.1))
            b = bias_variable([shape_11[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.pool4 , W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv5_1 = conv
            self.parameters += [W, b]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            shape_12=[3, 3, 512, 512]
            W = tf.Variable(tf.truncated_normal(shape_12,stddev=0.1))
            b = bias_variable([shape_12[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.conv5_1, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv5_2 = conv
            self.parameters += [W, b]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            shape_13=[3, 3, 512, 512]
            W = tf.Variable(tf.truncated_normal(shape_13,stddev=0.1))
            b = bias_variable([shape_13[3]])
            conv = tf.nn.conv2d(self.conv5_2, W, [1, 1, 1, 1], padding='SAME')+b
            self.conv5_3 = tf.nn.relu(conv, name=scope)
            self.parameters += [W, b]

        # pool5
        self.pool5 = max_pool(self.conv5_3)
        
    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i <= 25:
                sess.run(self.parameters[i].assign(weights[k]))

    def __init__(self, imgs, weights, sess):
        self.imgs = imgs
        self.imgs_update = tf.Variable(tf.constant(0.0, shape=[1,224,224,3], dtype=tf.float32))
        self.convlayers()
        self.load_weights(weights, sess)

    def neural_style(self, content_features, style_features, alpha, beta):

        M_1 = style_features[0].shape[1] * style_features[0].shape[2]
        N_1 = style_features[0].shape[3]
        G_1 = gram_matrix(style_features[0], M_1, N_1)
        A_1 = gram_matrix(self.conv1_1, M_1, N_1)
        contribution_1 = contribution_of_layer_to_loss(M_1,N_1, G_1, A_1)

        M_2 = style_features[1].shape[1] * style_features[1].shape[2]
        N_2 = style_features[1].shape[3]
        G_2 = gram_matrix(style_features[1], M_2, N_2)
        A_2 = gram_matrix(self.conv2_1, M_2, N_2)
        contribution_2 = contribution_of_layer_to_loss(M_2,N_2, G_2, A_2)

        M_3 = style_features[2].shape[1] * style_features[2].shape[2]
        N_3 = style_features[2].shape[3]
        G_3 = gram_matrix(style_features[2], M_3, N_3)
        A_3 = gram_matrix(self.conv3_1, M_3, N_3)
        contribution_3 = contribution_of_layer_to_loss(M_3,N_3, G_3, A_3)

        M_4 = style_features[3].shape[1] * style_features[3].shape[2]
        N_4 = style_features[3].shape[3]
        G_4 = gram_matrix(style_features[3], M_4, N_4)
        A_4 = gram_matrix(self.conv4_1, M_4, N_4)
        contribution_4 =contribution_of_layer_to_loss(M_4,N_4, G_4, A_4)

        M_5 = style_features[4].shape[1] * style_features[4].shape[2]
        N_5 = style_features[4].shape[3]
        G_5 = gram_matrix(style_features[4], M_5, N_5)
        A_5 = gram_matrix(self.conv5_1, M_5, N_5)
        contribution_5 = contribution_of_layer_to_loss(M_5,N_5, G_5, A_5)


        content_loss = tf.reduce_sum(tf.square(self.conv5_2 - content_features))
        style_loss = (contribution_1 + contribution_2 + contribution_3 + contribution_4 + contribution_5)/5
        self.loss = alpha*content_loss + beta*style_loss
        self.temp = set(tf.all_variables())
        self.optim = tf.train.AdamOptimizer(1)
        self.train_step = self.optim.minimize(self.loss, var_list=[self.imgs_update])



if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, '/home/rishabh/Downloads/GNR_COURSE_PROJECT/vgg16_weights.npz', sess)

    content_img = cv2.imread('//home/rishabh/Downloads/GNR_COURSE_PROJECT/input.jpg')
    content_img = cv2.resize(content_img, (224, 224))

    style_img = cv2.imread('/home/rishabh/Downloads/GNR_COURSE_PROJECT/style2.jpg')
    style_img = cv2.resize(style_img, (224, 224))

    style_assign = vgg.imgs_update.assign(np.asarray([style_img]).astype(float))
    sess.run(style_assign)
    style_features = [0 for i in range(5)]
    style_features = sess.run([vgg.conv1_1,vgg.conv2_1,vgg.conv3_1,vgg.conv4_1,vgg.conv5_1], feed_dict={vgg.imgs: [style_img]})

    content_assign = vgg.imgs_update.assign(np.asarray([content_img]).astype(float))
    sess.run(content_assign)
    content_features = sess.run(vgg.conv5_2, feed_dict={vgg.imgs: [content_img]})

    result_img = np.zeros((1,224,224,3)).tolist()
    
    vgg.neural_style(content_features, style_features, alpha, beta)

    sess.run(tf.variables_initializer(set(tf.all_variables()) - vgg.temp))

    for i in range(200):
        loss = sess.run(vgg.loss, feed_dict={vgg.imgs: result_img})
        print("iteration",i,"loss",loss)
        update = sess.run(vgg.train_step, feed_dict={vgg.imgs: result_img})

    result_img = sess.run(vgg.imgs_update, feed_dict={vgg.imgs: result_img})

    result = np.asarray(result_img[0]).astype(np.uint8)
    

    #imsave('output.jpg', x)
    cv2.imwrite( '/home/rishabh/Downloads/GNR_COURSE_PROJECT/output.jpg', result );
