---
title: "Reference"
date: 2018-01-28
tags: [machine learning, data science, neural network]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Machine Learning, Perceptron, Data Science"
mathjax: "true"
---

# H1 Heading

## H2 Heading

### H3 Heading

Here's some basic text.

And here's some *italics*

Here's some **bold** text.

What about a [link](https://github.com/dataoptimal)?

Here's a bulleted list:
* First item
+ Second item
- Third item

Here's a numbered list:
1. First
2. Second
3. Third

Active learning is a special case of machine learning in which a learning algorithm is able to interactively query the user (or some other information source called the oracle  to obtain the desired outputs at new data points. There are situations in which unlabeled data is abundant but manually labeling is expensive. In such a scenario, learning algorithms can actively query the user/teacher for labels. This type of iterative supervised learning is called active learning. Since the learner chooses the examples, the number of examples to learn a concept can often be much lower than the number required in normal supervised learning.








Python code block:
```python
    import numpy as np

    def test_function(x, y):
      z = np.sum(x,y)
      return z
```

R code block:
```r
library(tidyverse)
df <- read_csv("some_file.csv")
head(df)
```

Here's some inline code `x+y`.

Here's some math:

$$z=x+y$$

Latex Math

$$ \theta H = - \sum _ { x _ { = 0 } } ^ { 1 } \theta ^ { x } ( 1 - \theta ) ^ { 1 - x } \log \left[ \theta ^ { x } ( 1 - \theta ) ^ { 1 - x } \right] $$




<p align="center">
<img src="https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif">

</p>

<center>
*Fig. 2: The minimum dominating set of a graph*
</center>

Reference
Olsson, Fredrik (April 2009). "A literature survey of active machine learning in the context of natural language processing". SICS Technical Report T2009:06.
 Settles, Burr (2010). "Active Learning Literature Survey" (PDF). Computer Sciences Technical Report 1648. University of Wisconsin–Madison. Retrieved 2014-11-18.
