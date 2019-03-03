## TrAdaBoost

This is a Python implementation of TrAdaBoost proposed in [Boosting for Transfer Learning](http://www.cs.ust.hk/~qyang/Docs/2007/tradaboost.pdf).

I have tested the code on one domain adaptation dataset: [landmine](https://www.cse.ust.hk/TL/). However, the performance of TrAdaBoost seems to be the same as base learner SVM. 

On the other hand, I modified the way of update weights from line 47 to line 50 in `TrAdaBoost.py`.

Besides, `svm.py` is a self-implemented python version of algorithm SVM, which supports multiplying different coefficients which calculating loss. The accuracy of this code is similar to built-in function, while the speed is much much much slower. Actually, the function `fit` in `sklearn` also provides the same option. So I use built-in function in `TrAdaBoost.py` finally.

**TODO**: TrAdaBoost-TSVM


### Requirements
> python3.6, numpy, sklearn

### Acknowledgement
The C++ version source code is provided by the author in [https://www.cse.ust.hk/TL/](https://www.cse.ust.hk/TL/).