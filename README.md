# IMDB_text_classification
Classifying text from IMDB data using multi-layer RNN.

# Description
This project containt imdb text classification model in Tensorflow that uses multi-layer RNN.
Our model follows the Embed-Encode-Predict paradigm.

# Files
imdbNet contains data generator, imdb model, train class and other functions needed to train the imdb model.
imdb_demo.ipynb is a demo of how to use the files in the imdbNet folder.

# Comment
Dut to limited computational resource, we have trained the model using limited dataset, epochs and hidden units. Hence the accuracy was not good. Accuracy can be significantly improved using larger dataset, more epochs , more hidden units and larger embedding dimension.

# Future Improvement
Experiment with bidirectional RNN with attention.



