import data
import network
import sys

def b_part():
    training_data, test_data = data.load(train_size=10000, test_size=5000)
    net = network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data, plot_graphs=True)

def c_part():
    training_data, test_data = data.load(train_size=50000, test_size=10000)
    net = network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

def d_part():
    pass


if len(sys.argv) < 2:
    print "Please enter which part do you want to execute - b, c, d or all"
    exit()
cmds = sys.argv[1:]
for cmd in cmds:
    if cmd not in ['b', 'c', 'd', 'all']:
        print "Unknown argument %s. please run with b, c, d or all" % cmd
        exit()

if 'b' in cmds or 'all' in cmds:
    b_part()
if 'c' in cmds or 'all' in cmds:
    c_part()
if 'd' in cmds or 'all' in cmds:
    d_part()