import Training_Data_matrix as TDM
import numpy as np
import operator
import timeit

X_train = TDM.X_train
y_train = TDM.y_train
vocabulary_size = TDM.vocabulary_size

class RNNNumpy:

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))

    def softmax(self, x):
        xt = np.exp(x - np.max(x))
        return xt/np.sum(xt)
        RNNNumpy.softmax = softmax

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x) 
        #During forward prop we save all hidden states in s 
        #We add one additional element for the intial hidden state, which is set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        #The outputs at each time step
        o = np.zeros((T, self.word_dim))
        #For each time step
        for t in np.arange(T):
            #indexing U by x[t]
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = self.softmax(self.V.dot(s[t]))
        return [o, s]
        RNNNumpy.forward_propagation = forward_propagation

    def predict(self, x):
        #perform fwd prop and return index of highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)
        RNNNumpy.predict = predict

    def calculate_total_loss(self, x, y):
        L = 0
        #for each sentence
        #for i in np.arange(len(y)):
            #print i
            #print len(y)
            #print x
            #o, s = self.forward_propagation(x[i])
            #only prediction of the correct words
            #correct_word_predicitons = o[np.arange(len(y[i])), y[i]]
            #Add to the loss based on how off the predictions are
            #L += -1*np.sum(np.log(correct_word_predicitons))
        #return L
        return 0

    def calculate_loss(self, x, y):
        #Average loss
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y)/N
        RNNNumpy.calculate_total_loss = calculate_total_loss
        RNNNumpy.calculate_loss = calculate_loss

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])              
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]
        RNNNumpy.bptt = bptt

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)
        #RNNNumpy.gradient_check = gradient_check
        
    # Performs one step of SGD.
    def numpy_sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
        #RNNNumpy.numpy_sgd_step = numpy_sgd_step
        # Outer SGD Loop
        # - model: The RNN modedknsdklanlkdnlal instance
        # - X_train: The training data set
        # - y_train: The training data labels
        # - learning_rate: Initial learning rate for SGD
        # - nepoch: Number of times to iterate through the complete dataset
        # - evaluate_loss_after: Evaluate the loss after this many epochs
    def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
        # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                print "hihih"
                print X_train
                print X_train[1]
                loss = model.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5 
                    print "Setting learning rate to %f" % learning_rate
                sys.stdout.flush()
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                model.numpy_sgd_step(X_train[i], TDM.y_train[i], learning_rate)
                num_examples_seen += 1



np.random.seed(10)
model = RNNNumpy(TDM.vocabulary_size)
#o, s = model.forward_propagation(TDM.X_train[10])
#print o.shape
#print o

#predictions = model.predict(TDM.X_train[10])
#print predictions.shape
#print predictions
#print "Expected Loss for random predictions: %f" % np.log(TDM.vocabulary_size)
#print "Actual loss: %f" % model.calculate_loss(TDM.X_train[:1000], TDM.y_train[:1000])

# To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
#grad_check_vocab_size = 100
#np.random.seed(10)
#model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
#model.gradient_check([0,1,2,3], [1,2,3,4])

#timeit.timeit(model.numpy_sgd_step(X_train[10], y_train[10], 0.005))
#print X_train[:100]
#print model.forward_propagation(X_train[1])
losses = model.train_with_sgd(model, X_train[:100], nepoch=10, evaluate_loss_after=1)
