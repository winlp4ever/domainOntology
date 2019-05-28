# Towards Automatic Correction

__Relevant terms:__ #kddk (Knowledge discovery by guided domain knowledge) #ontology #topic-modeling

## Topic modeling by LDA

A highly correlated problem with ours, it is amongst the most classic problems in _NLP_. Until today, _Latent Dirichlet Allocation_ (LDA) is still a must-try when tackling this task. 

So, what is _topic modeling_? As simple as the name indicates, it is the task of **detecting possible topics** from a bunch of texts, possibly without any annotation. The task itself may be categorized as supervised or unsupervised problem, depending on whether the topics are prior known to us or not. The more difficult one is of course without prior knowledge about likely topics or keywords, which coheres much more with our problem settings.

Normally, a solution to a topic modeling task will always consist of transformation of texts to real dense vectors, which is the only way for us to detect the cohesion and dispersion (because we can only do that with maths!) and, a better term, the distribution of the data itself. However, at the end of the day, we only have in-hand a handful of vectors, not words, not phrases. Of course we can trace back to words whose representative vectors appear most consistent with the topic vectors. But most of the time, they do not well resume the topics and it's us who have to deduce those manually. So that's a huge drawback. How to solve this? Possibly with a deep complicated neural net (not so complicated if you already know about something like CNN or AutoEncoder) but we'll come to that later. For now, let's see what classic methods can get for us.

Now, how does LDA work? In a few words, it sees each word, each sentence, each document as a weighted sum of various topics. And what this method do is try to firstly assign each word to several topics that it may belong to, then view each document as a bag of words and deduce the topics highly present inside.

Let's dig in for more details, this time with maths:

### LDA

Assume there are $k$ topics in total, a vocabulary of size $V$, $M$ is the number of documents, document $i$ contains $N_i$ words. 
