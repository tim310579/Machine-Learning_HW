from numpy import exp, array, random, dot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
class NeuronLayer():
    def __init__(self, output_num, input_num):
        self.weights = np.random.randn(input_num, output_num)


class NeuralNetwork():
    def __init__(self, layer1, layer2, layer3, layer4):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, x_y, label, iters):
        for i in range(iters):
            
            output_layer_1, output_layer_2, output_layer_3, output_layer_4 = self.think(x_y)

            layer4_error = label - output_layer_4
            layer4_delta = layer4_error * self.sigmoid_derivative(output_layer_4)          
            
            layer3_error = layer4_delta.dot(self.layer4.weights.T)
            layer3_delta = layer3_error * self.sigmoid_derivative(output_layer_3)
            
            layer2_error = layer3_delta.dot(self.layer3.weights.T)
            layer2_delta = layer2_error * self.sigmoid_derivative(output_layer_2)

            layer1_error = layer2_delta.dot(self.layer2.weights.T)
            layer1_delta = layer1_error * self.sigmoid_derivative(output_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = x_y.T.dot(layer1_delta)
            layer2_adjustment = output_layer_1.T.dot(layer2_delta)
            layer3_adjustment = output_layer_2.T.dot(layer3_delta)
            layer4_adjustment = output_layer_3.T.dot(layer4_delta)
            # Adjust the weights.
            self.layer1.weights += layer1_adjustment
            self.layer2.weights += layer2_adjustment
            self.layer3.weights += layer3_adjustment
            self.layer4.weights += layer4_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_layer1 = self.sigmoid(dot(inputs, self.layer1.weights))
        output_layer2 = self.sigmoid(dot(output_layer1, self.layer2.weights))
        output_layer3 = self.sigmoid(dot(output_layer2, self.layer3.weights))
        output_layer4 = self.sigmoid(dot(output_layer3, self.layer4.weights))
        
        return output_layer1, output_layer2, output_layer3, output_layer4

    # The neural network prints its weights
    def print_weights(self):
        print ("    Layer 1 (4 neurons, each with 2 inputs): ")
        print (self.layer1.weights)
        print ("    Layer 2 (4 neurons, with 4 inputs):")
        print (self.layer2.weights)
        print ("    Layer 3 (4 neurons, with 4 inputs):")
        print (self.layer3.weights)
        print ("    Layer 4 (1 neuron, with 4 inputs):")
        print (self.layer4.weights)
def acc(pre, y):
    cor = 0
    for i in range(100):
        if(pre[i,0] >= 0.5):
            if(y[i,0]==1): cor += 1
            else: print(i,'lalal')
        else:
            if(y[i,0]==0): cor += 1
            else: print(i, 'lalala')
    return cor
def loss(pre, act):
    err = 0
    for i in range(len(pre)):
        err += (pre[i]-act[i])*(pre[i]-act[i])
    return err
if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    # Create layer 1 (4 neurons, each with 2 inputs)
    layer1 = NeuronLayer(4, 2)

    # Create layer 2 (4 neurons, with 4 inputs)
    layer2 = NeuronLayer(4, 4)

    # Create layer 3 (4 neurons, with 4 inputs)
    layer3 = NeuronLayer(4, 4)

    # Create layer 4 (1 neuron, with 4 inputs)
    layer4 = NeuronLayer(1, 4)
    
    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2, layer3, layer4)

    print ("Random starting weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    df = pd.read_csv('data.txt', header = None)
    df.columns = ['x', 'y', 'label']
    dfx = np.array([df['x']]).T
    dfy = np.array([df['y']]).T
    X = np.hstack((dfx, dfy))
    y = np.array([df['label']]).T
    X2 = X/np.amax(X, axis=0) # maximum of X array
    i = 1
    for i in range(1, 11):
        neural_network.train(X2, y, i*10000)
        print ("New weights after training %d times: " % (i*10000))
        neural_network.print_weights()

        hid1, hid2, hid3, out = neural_network.think(array(X2))
        los = loss(out, y)
    #print(los)
        print("epochs %d loss: %e" % (i*10000, los))
        print('Accuracy :', acc(out, y),'%')
    tmp_x1 = []
    tmp_y1 = []
    tmp_x0 = []
    tmp_y0 = []
    hid1, hid2, hid3, out = neural_network.think(array(X2))
    print(out)
    for i in range(100):
        data = X[i]
        hid1, hid2, hid3, out = neural_network.think(array(X2[i]))
        if(out > 0.5):
            tmp_x1.append(data[0])
            tmp_y1.append(data[1])
        else:
            tmp_x0.append(data[0])
            tmp_y0.append(data[1])
    plt.plot(tmp_x1, tmp_y1, 'o', color = 'b', label = '1')
    plt.plot(tmp_x0, tmp_y0, 'o', color = 'r', label = '0')
    plt.legend(loc = 'best')
    a = np.arange(0, 100)
    b = a
    plt.plot(a, b, c = 'g')
    plt.title(u'Predict result',fontsize=17)
    plt.savefig('predict.png')
    plt.close('all')
    
    real_x1 = []
    real_y1 = []
    real_x0 = []
    real_y0 = []
    
    for i in range(100):
        data = X[i]
        if(data[0] < data[1]):          #x < y label 1
            real_x1.append(data[0])
            real_y1.append(data[1])
        else:
            real_x0.append(data[0])
            real_y0.append(data[1])
    plt.plot(real_x1, real_y1, 'o', color = 'b', label = '1')
    plt.plot(real_x0, real_y0, 'o', color = 'r', label = '0')
    plt.legend(loc = 'best')
    a = np.arange(0, 100)
    b = a
    plt.plot(a, b, c = 'g')
    plt.title(u'Ground truth',fontsize=17)
    plt.savefig('truth.png')
    plt.close('all')
        
