# End-2-end AutoML Pipelines

## Abstract
Over the last decade, deep learning has gained traction in the field of Computer Science. Neural networks-based models are being leveraged to perform many tasks that once required human intervention.  Thus, it is no surprise that the number of new architectures, the optimisers and hyperparameters being explored is ever-increasing. In such a scenario, it can often become increasingly difficult to find the right combination for the task at hand. The problem of this domain is no longer the unavailability of robust models, but rather the use of the best of the available ones. One solution to the problem of this time-consuming task has been answered by the discovery of AutoML. Through this project, we present our own implementation of an end-to-end AutoML pipeline. Using the underlying concepts of HML Opt and Differentiable Architecture Search (DARTS), we build a system that takes in raw data as input and uses a probabilistic machine learning algorithm to determine the best deep learning model parameters for that task.
Keywords: AutoML, Neural Architecture Search , Deep Learning

## Code
The code is organized into modules

### pdarts
Here you can find our re-implementation of the PDARTS paper with some cleanup, and the necessary changes to make it compatible with AutoGOAL


### preprocessing
Here we define the grammar of the allowed preprocessing steps.

We expose this to allow AutoGOAL to explore the grammar correctly and generate valid pieplines


### pipeline
Here we define the overall system, we create a class that when given to AutoGOAL will generate valid end-2-end pipelines

