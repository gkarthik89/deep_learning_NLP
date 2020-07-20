# Deep Learning Natural Language Processing.
This project classifies text into pre-defined categories using deep learning neural networks.

# DataSet
The dataset can be csv file with the following fields:
 <ol>
 <li> ID -> Unique IdentifierID </li>
 <li> LabelID -> Integer, representing category ID.</li>
 <li> LabelName -> Text, representing the category names.</li>
 <li> Notes -> Text, Sentences/notes representing each category.</li>
</ol>

Please create your own dataset. There are a lot of dataset available online that can be used.

# Pre-Processing and Text Cleaning.
<h4>NLTK – Natural Language Toolkit: </h4>
<p>
NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries.</br>
<b>Stop Words:</b> A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore, both when indexing entries for searching and when retrieving them as the result of a search query. We would not want these words to take up space in our database or taking up valuable processing time. NLTK (Natural Language Toolkit) in python has a list of stopwords stored in 16 different languages. </p>

<h4>Beautiful Soup:</h4>
<p>Beautiful Soup is a library that makes it easy to scrape information from web pages. It sits atop an HTML or XML parser, providing Pythonic idioms for iterating, searching, and modifying the parse tree.</p>

<h4>Keras Text processing library:</h4>
<p>Text tokenization utility class allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, based on word count, based on tf-idf.</p>

<ol>
<li>The case notes were stripped of special characters like ` [/(){}\[\]\|@,;.#+_=*&` and replaced with space.</li>
<li>All the words in the notes were converted into lower case.</li>
<li>Using BeautifulSoup library the HTML tags were escaped and decoded.</li>
<li>The stop words in every note was removed to achieve better accuracy.</li>
<li>The notes were then tokenized using the keras text preprocessing library.</li>
<li>The texts were then converted to matrix to be used for modeling.</li>
<li>The labels were then encoded to convert into numerics using the labelEncoder library from sklearn.</li>
<li>The labels were further hot-encoded using keras.</li>
</ol>

# Neural network (DNN) for text classification:
<h3>Model:</h3>
<h4>Keras:</h4>
<p>Keras is an open-source neural-network library written in Python. It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, R, Theano, or PlaidML. Designed to enable fast experimentation with deep neural networks, it focuses on being user-friendly, modular, and extensible. It was developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System).
Keras contains numerous implementations of commonly used neural-network building blocks such as layers, objectives, activation functions, optimizers, and a host of tools to make working with image and text data easier to simplify the coding necessary for writing deep neural network code.
The Model is the core Keras data structure. There are two main types of models available in Keras: the Sequential model, and the Model class used with the functional API.
The Sequential model is a linear stack of layers, and the layers can be described very simply. Each layer definition requires one line of code, the compilation (learning process definition) takes one line of code, and fitting (training), evaluating (calculating the losses and metrics), and predicting outputs from the trained model each take one line of code.</p>
<h4>TensorFlow:</h4>
<p>TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.
 Created by the Google Brain team, TensorFlow is an open source library for numerical computation and large-scale machine learning. TensorFlow bundles together a slew of machine learning and deep learning (aka neural networking) models and algorithms and makes them useful by way of a common metaphor. It uses Python to provide a convenient front-end API for building applications with the framework, while executing those applications in high-performance C++.</p>

<h4>Hyper Parameter Tuning:</h4>
<ol>
<li>Activation function: Sigmoid, softmax</li>
<li>Dropout: 0.7</li>
<li>Number of Nodes: 2000</li>
<li>Optimizer: adam</li>
<li>Loss: categorical cross entropy</li>
<li>Metrics: accuracy.</li>
</ol>

# Predictions:
 The predictions are done using a holdout dataset. The trained labels and saved models are loaded to perform the predictions.
