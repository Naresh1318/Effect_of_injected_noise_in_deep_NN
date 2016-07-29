import mnist_loader
import network2
import numpy as np


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

eta = 0.9
m_b_s = 10
epochs = 30
trials = 10
trial_ev = []

for t in xrange(trials):
    net = network2.Network([784, 50, 50, 50, 50, 10], cost=network2.CrossEntropyCost)
    net.default_weight_initializer()
    _,ev,_,_ = net.SGD(training_data[:1000], epochs, m_b_s, eta, evaluation_data=test_data[:1000],monitor_evaluation_accuracy=True)
    print "Trial {} Complete".format(t + 1)
    print "Maximum Evaluation Accuracy : {}".format(np.amax(ev))
    trial_ev.append(np.amax(ev))

Avg_ev = np.mean(trial_ev)
Max_ev = np.amax(trial_ev)
print "Average Evaluation Accuracy for {} trials is {}".format(trials,Avg_ev)
print "Maximum Evaluation Accuracy for {} trials is {}".format(trials,Max_ev)