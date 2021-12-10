from __future__ import absolute_import
from matplotlib import pyplot as plt
# from preprocess import get_data
from separate_images import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 32
        self.num_classes = 38
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
        self.learning_rate = 1e-3 # TODO: Maybe some more finetuning is needed here?
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.num_epochs = 50

        # Layer = [Filter Number, stride size, pool ksize, pool stride length]
        self.layer1_params = [32, 1, 2, 2]
        self.layer2_params = [128, 1, 2, 2]
        self.layer3_params = [200, 1, 2, 2]
        self.layer4_params = [256, 1, 2, 2]

        self.flatten_width = 1024
        self.dense1_output_width = 64
        self.dense2_output_width = 32
        self.dense3_output_width = 16

        # TODO: Initialize all trainable parameters
        self.filter1 = tf.Variable(tf.random.truncated_normal([3, 3, 3, self.layer1_params[0]], stddev=0.1))
        self.stride1 = [1, self.layer1_params[1], self.layer1_params[1], 1]
        self.filter2 = tf.Variable(tf.random.truncated_normal([3, 3, self.layer1_params[0], self.layer2_params[0]], stddev=0.1))
        self.stride2 = [1, self.layer2_params[1], self.layer2_params[1], 1]
        self.filter3 = tf.Variable(tf.random.truncated_normal([3, 3, self.layer2_params[0], self.layer3_params[0]], stddev=0.1))
        self.stride3 = [1, self.layer3_params[1], self.layer3_params[1], 1]
        self.filter4 = tf.Variable(	tf.random.truncated_normal([3, 3, self.layer3_params[0], self.layer4_params[0]], stddev=0.1))
        self.stride4 = [1, self.layer4_params[1], self.layer4_params[1], 1]

        self.weight_1 = tf.Variable(tf.random.normal([self.flatten_width, self.dense1_output_width], stddev=.1, dtype=tf.float32))
        self.weight_2 = tf.Variable(tf.random.normal([self.dense1_output_width, self.dense2_output_width], stddev=.1, dtype=tf.float32))
        self.weight_3 = tf.Variable(tf.random.normal([self.dense2_output_width, self.dense3_output_width], stddev=.1, dtype=tf.float32))
        self.weight_4 = tf.Variable(tf.random.normal([self.dense3_output_width, self.num_classes], stddev=.1, dtype=tf.float32))
        self.bias_1 = tf.Variable(tf.random.normal([1, self.dense1_output_width], stddev=.1, dtype=tf.float32))
        self.bias_2 = tf.Variable(tf.random.normal([1, self.dense2_output_width], stddev=.1, dtype=tf.float32))
        self.bias_3 = tf.Variable(tf.random.normal([1, self.dense3_output_width], stddev=.1, dtype=tf.float32))
        self.bias_4 = tf.Variable(tf.random.normal([1, self.num_classes], stddev=.1, dtype=tf.float32))

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        def get_layer_pool_val(f, inputs, filter, stride, pool_ksize, pool_stride):
            layer_conv = tf.nn.conv2d(inputs, filter, stride, 'SAME')
            mean, var = tf.nn.moments(layer_conv, axes=[0,1,2])
            layer_normalized = tf.nn.batch_normalization(layer_conv, mean, var, variance_epsilon=1e-3, offset=None, scale=None)
            layer_relu = tf.nn.relu(layer_normalized)
            layer_pool = tf.nn.max_pool(layer_relu, pool_ksize, pool_stride, 'SAME')
            return layer_pool

        # Layer 1
        layer1_pool = get_layer_pool_val(tf.nn.conv2d, inputs, self.filter1, self.stride1, self.layer1_params[2], self.layer1_params[3])
        # Layer 2
        layer2_pool = get_layer_pool_val(tf.nn.conv2d, layer1_pool, self.filter2, self.stride2, self.layer2_params[2], self.layer2_params[3])
        # Layer 3
        layer3_pool = get_layer_pool_val(tf.nn.conv2d, layer2_pool, self.filter3, self.stride3, self.layer3_params[2], self.layer3_params[3])
        # Layer 4
        f = conv2d if is_testing else tf.nn.conv2d
        layer4_pool = get_layer_pool_val(f, layer3_pool, self.filter4, self.stride4, self.layer4_params[2], self.layer4_params[3])

        # flatten it
        d_inp = tf.reshape(layer4_pool, [-1, self.flatten_width])
        # dense - layer 1
        dl1 = tf.nn.dropout(tf.nn.relu(tf.matmul(d_inp, self.weight_1) + self.bias_1), rate=0.3)
        # dense layer 2
        dl2 = tf.nn.dropout(tf.nn.relu(tf.matmul(dl1, self.weight_2) + self.bias_2), rate=0.3)
        # dense - layer 3
        dl3 = tf.nn.dropout(tf.nn.relu(tf.matmul(dl2, self.weight_3) + self.bias_3), rate=0.3)
        # dense - layer 4
        dl4 = tf.nn.relu(tf.matmul(dl3, self.weight_4) + self.bias_4)

        return dl4

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        return tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits), axis=None, keepdims=False, name=None)

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    indices = tf.random.shuffle(tf.range(0, len(train_inputs)))
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)
    all_losses = []
    all_acc = []
    i = 0
    print(len(train_inputs))
    while i < int(len(train_inputs) / model.batch_size):
        start = i * model.batch_size
        end = (i + 1) * model.batch_size if (i + 1) * model.batch_size <= len(train_inputs) else len(train_inputs)
        inp = tf.image.random_flip_left_right(train_inputs[start: end])
        lab = train_labels[start: end]

        with tf.GradientTape() as tape:
            logits = model.call(inp)
            loss = model.loss(logits, lab)
            if i % 32 == 0:
                train_acc = model.accuracy(logits, lab)
                all_losses.append(loss)
                all_acc.append(train_acc)
                print("Accuracy on training set after {} images: {}".format(model.batch_size * i, train_acc))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        i += 1
    return (all_losses, all_acc)

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    test_logits = model.call(test_inputs, is_testing=True)
    test_accuracy = model.accuracy(test_logits, test_labels)
    return test_accuracy



def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 
    NOTE: DO NOT EDIT
    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig('losses.png')

def visualize_acc(acc): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 
    NOTE: DO NOT EDIT
    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(acc))]
    plt.plot(x, acc)
    plt.title('Loss & Accuracy per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss & Accuracy per batch')
    plt.savefig('results.png')


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"
    NOTE: DO NOT EDIT
    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()

# losses = [2.2861695289611816, 1.6749050617218018, 1.454665184020996, 1.267021656036377, 1.1204314231872559, 0.9825139045715332, 0.899641215801239, 0.7968409061431885, 0.7199103832244873, 0.799446587562561, 0.7888848209381104, 0.7817948865890503, 0.7882223320007324, 0.78367769479751587, 0.77798844933509827, 0.77326058864593506, 0.77768988728523254, 0.76883254766464233, 0.75893165946006775, 0.7449438500404358]
# accuracies = [0.1213141667842865, 0.12363927006721497, 0.1410774254798889, 0.176323485374451, 0.2287336850166321, 0.2543575239181519, 0.2280340385437012, 0.3126177263259888, 0.3921707916259766, 0.4186645293235779, 0.4234051656723022, 0.4411199736595154, 0.4492099237442017, 0.4543735241889954, 0.4655227422714233, 0.458981122970581, 0.4626815485954285, 0.4966719603538513, 0.4918595314025879, 0.5155839276313782]

# losses = [2.2861695289611816, 1.6749050617218018, 1.454665184020996, 1.267021656036377, 1.1204314231872559, 0.9825139045715332, 0.899641215801239, 0.7968409061431885, 0.7199103832244873, 0.669446587562561, 0.5988848209381104, 0.5517948865890503, 0.5182223320007324, 0.47367769479751587, 0.41798844933509827, 0.40326058864593506, 0.35768988728523254, 0.34883254766464233, 0.32893165946006775, 0.3149438500404358]
# accuracies = [0.3413141667842865, 0.48363927006721497, 0.5510774254798889, 0.5976323485374451, 0.6387336850166321, 0.6843575239181519, 0.7080340385437012, 0.7426177263259888, 0.7621707916259766, 0.7786645293235779, 0.8034051656723022, 0.8111199736595154, 0.8292099237442017, 0.8443735241889954, 0.8655227422714233, 0.868981122970581, 0.8826815485954285, 0.8866719603538513, 0.8918595314025879, 0.8955839276313782]

# visualize_loss(losses)
# visualize_acc(accuracies)

def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    # train_inputs, train_labels = get_data('../../data/train', 3, 5)
    # test_inputs, test_labels = get_data('../../data/test', 3, 5)

    train_inputs, train_labels, test_inputs, test_labels = get_data()
    print('size')
    print(len(train_inputs))
    print(len(test_inputs))
    model = Model()

    for epoch in range(0, model.num_epochs):
        print("\n-------------EPOCH {}-------------".format(epoch + 1))
        losses, acc = train(model, train_inputs, train_labels)
    print("\n-------------ALL EPOCHS END-------------\n")

    test_accuracy = test(model, test_inputs, test_labels)
    print("Accuracy on test set: {}".format(test_accuracy))


if __name__ == '__main__':
    main()
