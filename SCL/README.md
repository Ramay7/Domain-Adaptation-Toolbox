# SCL-MI (Structual Correspondence Learning with Mutual Information)

This is a Python implementation of SCL-MI proposed in [Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification, ACL'07](https://www.cs.jhu.edu/~mdredze/publications/sentiment_acl07.pdf).

## Requirements
> python 3.6
numpy
sklearn
pickle
matplotlib

## Results

![](https://github.com/Ramay7/Domain-Adaptation-Toolbox/blob/master/SCL/result.png)

Since I do not find the source code by authors, I cannot know the way of feature representation used in authors' experiments. I have tried two kinds of feature representation in NLP tasks, however, the accuracy of both solutions was lower than that reported in original paper. 

What's more, it seems that the number of pivot features has no influence in the second way of feature representation. Due to limited time, this project can only go so far.

And I would thank a lot if you find some bugs in the project, and be willing to see the corresponding issues.

## Acknowledgement

The code is based on [https://github.com/yftah89/structural-correspondence-learning-SCL](https://github.com/yftah89/structural-correspondence-learning-SCL).
