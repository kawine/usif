# uSIF 

This is an implementation of *unsupervised smoothed inverse frequency* (uSIF), a simple but effective way to create sentence embeddings without any labelled data (Best Paper, Repl4NLP @ ACL 2018). See the [paper](http://aclweb.org/anthology/W18-3012) for more details.

1. Unzip the pre-trained ParaNMT word vectors (thanks to John Wieting for providing this).
1. Install the necessary python languages.
1. Initialize a uSIF embedding model with usif.py. `get_paranmt_usif` returns the model that uses the ParaNMT vectors.
