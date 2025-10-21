import numpy as np
#purpose of creation of this neural netwrok is to train MNIST image data to get right predictions
import struct
file_location = r"C:\Users\Apurv (AKM)\python\Deep Learning\Data\archive(2)\train-images.idx3-ubyte"
def get_train_image_info(filename):
	with open(filename, 'rb') as file:
		metadata = file.read(16)
		magic,number_of_images,rows,columns = struct.unpack('>iiii', metadata )
		image_bin_data = file.read()
		image_array = np.frombuffer(image_bin_data , dtype='uint-8')
		image = image_array.reshape(number_of_images,rows,columns)
		return image,number_of_images

def get_train_label(filename):
	with open(filename, 'rb') as file:
		metadata = file.read(8)
		magic,num_of_labels = struct.unpack('>ii', metadata)
		image_bin_data = file.read()
		image_array = np.frombuffer(image_bin_data, dtype='uint-8')
		image_label = image_array.reshape(num_of_labels)
		return image_label


#now I am going to try to integrate input with the things we have in neural network


target = np.zeros((output_nodes), dtype='float64')

label_data = get_train_label(r"C:\Users\Apurv (AKM)\python\Deep Learning\Data\archive(2)\train-labels.idx1-ubyte")
for i in label_data:
	target[i] = 1

training_image_data = get_train_image_info(file_location)


#I need to make a good user interface. I need to ask user the question whether they are going to train or test the neural network
#Also I need to make the Input layer flexible with the user data input.
#one more thing I don't want user to decide the output node instead I want it to predict the output node by itself
#getting input nodes
input_nodes = training_image_data.shape[1]*training_image_data[2]


#getting output nodes
unique_labels = np.unique(image_label)
output_nodes = np.unique(unique_labels)



num_hlayer = int(input("number of hidden layers = "))
num_nodes = int(input("enter number of nodes in each hidden layer = "))
# output_nodes = int(input("number of nodes in the output layer: "))
#--------------------------formation of node arrays
#creation of blank hidden neural network structure
node = np.zeros((num_hlayer,num_nodes))
weight = np.zeros((num_hlayer-1, num_nodes, num_nodes))
bias = np.zeros((num_hlayer-1, num_nodes, num_nodes))











#-------------------------------creation of input and output nodes
#input
# input_nodes = int(input('enter number of input nodes: '))
input_layer = np.zeros((input_nodes) , dtype = 'float64')
input_bias = np.zeros((input_nodes, num_nodes), dtype = 'float64')
input_weight = np.ones((input_nodes, num_nodes), dtype = 'float64')
input_hlayer = node[0, :]
[i,j] = [0,0]
while(i < input_nodes):
	j = 0
	while(j < num_nodes):
		temp = input_layer[i]
		temp = temp*input_weight[i,j]
		temp = temp + input_bias[i,j]
		input_hlayer[j] = input_hlayer[j] + temp

		j+=1

	i+=1

node[0,:] = input_hlayer

#-------------------------------making neural network hidden structure to interact with each other

[i,j,k] = [0,0,0]

while (i < num_hlayer - 1):
	j=0
	while(j < num_nodes):
		k=0
		while(k < num_nodes):
			if (k == 0):
				node[i,j] = 1/(1+np.exp(-1*node[i,j]))   #np.log(1 + np.exp(node[i,j]))
				temp = node[i, j]
				temp = weight[i,j,k]*temp
				temp = bias[i,j,k] + temp
			else:
				temp = node[i, j]
				temp = weight[i,j,k]*temp
				temp = bias[i,j,k] + temp

				node[i+1 , k] = node[i+1 , k] + temp
			k+=1

		j +=1
	i+=1



#after writing till here I have to now make input and output nodes which will be used to fetch data and give output

#output
output_layer = np.zeros((output_nodes) , dtype = 'float64')
output_bias = np.zeros((num_nodes, output_nodes), dtype = 'float64')
output_weight = np.zeros((num_nodes, output_nodes), dtype = 'float64')
output_hlayer = node[num_hlayer-1, :]

[i,j] = [0,0]
while(i < num_nodes):
	j=0
	while(j < output_nodes):
		if(j == 0):
			output_hlayer[i] = 1/(1+np.exp(-1*output_hlayer[i])) #np.log(1 + np.exp(output_hlayer[i])) 
			temp = temp*output_weight[i,j]
			temp = temp + output_bias[i,j]
			output_layer[j] = output_layer[j] + temp
		else:
			temp = output_hlayer[i]
			temp = temp*output_weight[i,j]
			temp = temp + output_bias[i,j]
			output_layer[j] = output_layer[j] + temp

		j+=1

	i+=1
output_layer = 1/(1+np.exp(-1*output_layer))


#---------------------------------------------------------------backpropagation
#get observed data
array = np.zeros((output_nodes), dtype = 'float64')
array = (array - output_layer)*(-2) # derivative of squared residual

def sigmoid(array_):
	array_ = 1/(1+np.exp(-1*array_))
	return array_

def derivative_sigmoid(array_):
	mul = 1/(1+np.exp(-1*array_))
	return mul*(1-mul)
#here I am only going to apply backprop to hidden layer and output layer rest input layer will be unaffected

#---------------------------------------------------------------backpropagation
#here to do backpropagation we have two methods one is to go back path by path compute the derivative of each path and then add
#the other method is that we go backwards but this time we use δ which basically says how much error influence is there.
#now each hidden node in any hidden layer is connected by every node in the previous hidden layer(by backpropagation context remember going backwards)
#so we compute δ for the current hidden layer for which I want to compute error influence. This is done by adding product of error influence of
#each previous hidden layer and weight connecting previous hidden layer nodes with current hidden layer node.
#after doing addition we multiply it by the derivation of hidden layer.
#This if we use an imaginary pointer we will find that that pointer is now pointing to the input of the current hidden node.
#
#
#
#
# observed data (true labels)
error = (output_layer - target)  # error at output

# learning rate
lr = 0.01

# derivative arrays for output and hidden layers
delta_output = error * derivative_sigmoid(output_layer)  # dL/dz for output layer

# Backprop from output layer to last hidden layer
for i in range(num_nodes):  # node in last hidden layer
    for j in range(output_nodes):  # node in output layer
        grad_w = delta_output[j] * node[num_hlayer-1, i]  # dL/dw = delta * activation
        grad_b = delta_output[j]
        # Update output weights and biases
        output_weight[i, j] -= lr * grad_w
        output_bias[i, j]   -= lr * grad_b

# Backprop through hidden layers (layer by layer)
# We'll compute delta for each hidden node and propagate backwards
delta_hidden = np.zeros((num_hlayer, num_nodes))

# Start with last hidden layer (l = num_hlayer - 1)
for i in range(num_nodes): #here i refers to the counting of the nodes in current hidden layer
    s = 0
    for j in range(output_nodes): #here j refers to the counting of the nodes in output layer
        s += delta_output[j] * output_weight[i, j]
    delta_hidden[num_hlayer-1, i] = s * derivative_sigmoid(node[num_hlayer-1, i])

# Backprop for hidden layers and update weights
for l in range(num_hlayer-2, -1, -1):  # l = num_hlayer-2 down to 0 #l points to the layer
    for i in range(num_nodes): #i pints to each node in current layer
        s = 0
        for k in range(num_nodes): #k points to node in layer previous to the current layer
            s += delta_hidden[l+1, k] * weight[l, i, k]
        delta_hidden[l, i] = s * derivative_sigmoid(node[l, i])
        # Update weights between layer l and l+1
        for k in range(num_nodes):
            grad_w = delta_hidden[l+1, k] * node[l, i]
            grad_b = delta_hidden[l+1, k]
            weight[l, i, k] -= lr * grad_w
            bias[l, i, k]   -= lr * grad_b

			

