''' Source : https://www.youtube.com/watch?v=p69khggr1Jo&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3&index=3 '''

'14th October 2019'
from numpy import exp,array, random,dot

class NeuralNetwork():
    def __init__(self):

        #seed the random numnber generator
        random.seed(1)

        self.synaptic_weights = 2 * random.random((3,1)) -1

    # The signmoid function which describes an s shaped curve
    def __sigmoid(self,x):
        return(1/(1+exp(-x)))


    def predict(self,inputs):
        # pass inputs through our neural network (one single neuron)
        return self.__sigmoid(self,dot(inputs,self.synaptic_weights))

    def train(self,training_set_inputs,training_set_outputs,number_of_training_iterations):

        for iteration in number_of_training_iterations:

            output = self.predict(self,training_set_inputs)
            error = training_set_outputs -  output

            adjustments = dot()




if __name__ = '__main__':

    #initialise a single neuron neual network
    neural_network =  NeuralNetwork()

    print('Random starting synaptic weights')
    print(neural_network.synaptic_weights)

    #The training set . We have 4 examples, each consisting of 3 input values nand 1 output value
    training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T

    #train the neural network using a training set .
    # Do it 10k times and make small adjustments each time

    neural_network.train(training_set_inputs,training_set_outputs,10000)

    print('New synaptic weight after training')
    print(neural_network.synaptic_weights)

    #Test the neural network with anew situation
    print("Considering new situation [1,0,0] -> ?")
    print(neural_network.think(array([1,0,0])))
