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
