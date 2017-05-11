**TensorFlow Basics Tutorial**

This repo shows some basic coding involved in implementing a neural network in tensorflow.
It contains two pieces of code:

* ```tfbasics.py```
    * **Short Context** : Entire code in Tensorflow is executed in a session. So untill we ```run``` our session via the ```tf.Session.run()``` command, our code won't show the mystical powers it possess. In short we just construct the graph of our network and no execution is performed.
    * **About Code** : The ```tfbasics.py``` is solely made to show how to execute our code in _tensorflow_ or in other words how to _run our session_. It contains an oversimplified example of multiplying two constants using ```tf.mul(x1, x2)``` where ```x1``` & ```x2``` are our two constant instantiations. The notable point here is that our ```tf.mul()``` won't actually compute the product untill we run our session which is being done via
    ```python
    with tf.Session() as sess:
        print sess.run(result)
    ```
    It is only when we call ```sess.run()``` that the product is computed and is being stored in ```result```.

* ```deep-net.py```
    * **Short Context** : A neural network in general, has three kinds of layers: *Input Layer*, *Hidden Layer* & *Output Layer*. A Neural Network is classified as ```deep``` or not based on the number of hidden layers it contains between the Input and Output Layer. A Neural Network has some very crucial components as described below:
        1. **Weight Matrix** : Neural Networks are all about extensive Matrix Algebra. The _Weight Matrix_ mentioned here is something that relates one layer to the other Layer. How you say?? Hmm...Let's see. Basically what a Neural Network does is it learns a function to generalize the relation between given Input & Output. So just like we see _**y = mx + c**_ we will have _**O** = **f(H<sup>(n)</sup>)**_ where **O** is output vector and **H<sup>(n)</sup>** is nth hidden layer vector which when chained down the 1<sup>st</sup> hidden layer bears a relation something like _**H<sup>(1)</sup> = g(I)**_ where **I** is the input layer. Very rarely this relation is linear in nature. This complexity is caught by the **Weight Matrix**.
        2. **Activation Function & Bias** : Following the analogy of how neurons in our brain function, _Activation Functions_ are the entities that turn our neuron **ON/OFF**. For a simplified task such as that of Binary Classification our _Activation Functions_ will have binary state, but in practical applications they are _nonlinear_ in nature that gives a lot of power and flexibilty to our network. We will talk about *Bias* a bit later.
        3. **Feed Forward, Back Propagation & Epoch** : _Feed Forward_ is nothing but running our network in the forward direction. How to run you say?? Hmm...Well you have the input vector, initial weight matrices (that are to be learnt obviously), the activation function and the bias. So all you have to do is put the different pieces of puzzles together (make the TF graph) and get your output vector (by running your session)! Now Backprop is this entire process in the reverse direction. It seems pretty straight forward eh ! Well theoretically it is, but when we proceed to the implementation phase, a lot of rigorous Matrix Algebra and Calculus is involved. It may be possible that at a point we come across an underivable/undefined expression. To safegaurd ourselves from such situations, we add *Bias* to our layers so that they won't vanish in between the process, so that won't be undefined during the process. Now _**1 Feed Foward** combined_ with _**1 Back Prop**_ make one **Epoch**.
        4. **Loss Function** : After running an _Epoch_, theoretically we shouldn't have any difference between the initial and final values but in real life, we do. We calculate this deviation and find how much does our value differ from the expected output value. How to minimize this?? Remember we had those *Weight Matrices* from before? We change the weights there so that we converge more and more towards our expected value. This is exactly what our loss function does. It's only job is to minimize this deviation or _loss_ and make our learn the mapping from input to output.
    * **About Code** : Now that we have gone through the various crucial **"cells" or "neurons"** that make our neural network, it should be fairly easy to follow along with the code. Choosing a very easy task and what is called the **"Hello World"** of Machine Learning, we will work with the **MNIST Dataset** (which contains large collection of hand written digits) to identify a letter given a test image. Here is the flow of the code explained:
        1. **Loading Dataset**: Tensorflow provides us with the dataset, so we don't have to download it from anywhere else. We first load the dataset via the line
        ```python
        from tensorflow.examples.tutorials.mnist import input_data
        ```
        and store it in one hot vector form via
        ```python
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        ```
        2. **Declaring Hyper-parameters (Constants that we need to define for our NN)** : This snippet contains all the hyper parameters for our network.
        ```python
        learning_rate = 0.001   # the rate at which we try to move while optimization
        hm_epochs = 10  # total number of epochs

        n_nodes_hl1 = 500  # number of nodes in hidden layer 1
        n_nodes_hl2 = 500  # number of nodes in hidden layer 2
        n_nodes_hl3 = 500  # number of nodes in hidden layer 3

        n_classes = 10  # As there are 10 possible outcomes (1, 2, 3...10)
        batch_size = 100  # We will read data in chunks, this will make our code a lot faster

        # ==========This the input layer declaration=============

        # input height x weight
        x = tf.placeholder('float', [None, 784])
        y = tf.placeholder('float')
        ```
        We can use ```tf.placeholder()``` to get a simple variable that can be assigned some value at a later stage.
        3. **Declaration of the Neural Network** : The function ```neural_network_model()``` contains our complete NN model. Here notice the declaration of hidden layers (```l1```, ```l2```, ```l3```) & output layer (```output```). It's kind of self explantory if we see their ```weights``` attribute we can find the pattern and map how we will proceed during a *Forward Feed* and *Back Prop*. It's nothing but a series of Matrix Multipications.
        We have used *Rectified Linear* as our *Activation Function* as can be seen here
        ```python
        tf.nn.relu(l2)  # activation function for hidden layer 2
        ```
        4. **Training the network** : This is the part where all optimizations will be performed. First we get our expected output from the network using ```prediction = neural_network_model(x)```, then we calculate the loss using ```softmax_cross_entropy_with_logits()``` and use ```AdamOptimizer``` to adjust our weights so as to learn the appropriate mapping of input to output as here
        ```python
        prediction = neural_network_model(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        ```

        Remember that we were declaring the graph of Neural Network until now. Using the ```tf.Session.run(tf.initialize_all_variables())``` we will trigger all our nodes in the Graph as active and finally performing the task ```hm_epochs``` times, we get out our final result.

For the implementation of some more complex neural networks like **RNN (Recursive Neural Network)** and **CNN (Convolutional Neural Network)** you can check out [this repo][!https://github.com/Demfier/sentiment_analysis]
