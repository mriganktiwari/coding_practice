# Lessons learnt from Karpathy videos:

## Video 1
#### Bi-gram model
1. **Zip** - to generate bigrams from words / text & iterate through them simultaneously
2. Adding start and end tokens
3. Use of **set** class constructor - throws away all duplicates
4. Creating mappings for **characters -> integers** & **integers -> characters**
5. Plot a numpy array of shape (28,28) with ```plt.imshow()```
	- Plots with texts into each grids (defined by x,y co-ordinates)
6. ```torch.multinomial```: samples the indexes from probability distribution, __num_samples__ times as per probability distribution
7. ```torch.Generator().manual_seed(2147483647)```: this generator object can be passed as ```generator=g``` argument several pytorch functions
8. ```.item()```: for a 1D array in torch gives the scalar value
9. Any operation along an **axis**: say we perform a ```.sum(axis=1)```
	- This would sum a row.
	- Hence, the axis 1 (# of columns) would squash down to 1.
	- **keepdim**: if True, the axis 1 dim is kept; else it'll be dropped.
10. Maximum likelihood estimation:
	- GOAL: __maximize likelihood__ of the data w.r.t. model parameters (statistical modeling).
	- equivalent to __maximum conditional probability__ - of sequence of chars occuring together (0 to 1).
	- equivalent to maximizing the __log likelihood__ (because log is monotonic) (-inf to 0).
	- equivalent to minimizing the __negative log likelihood__ (0 to inf).
	- equivalent to minimizing the __average negative log likelihood__.
11. Model smoothing:
	- In bi-gram model: pairs that never occured together in a dataset, will have their count = 0 in count matrix. This would introduce ```-inf``` in ```logprob```.
		- To handle such instances, 1 is added to all counts in the matrix

#### Neural network
1. ```torch.tensor``` should be preferred over ```torch.Tensor```
	- coz, later by default assigns ```dtype``` as ```float```
2. ```F.one_hot```
3. ```@``` is the matrix multiplication operator; ```*``` is the elementwise multiplication operator for tensors
4. To make sense of the numbers that come out of last layer, and interpret them as probabilities, we do below sequence of operations:
	- Think of these numbers as log-counts
		- ```logits = xenc @ W # log-counts```
	- Take exponential to convert them as counts
		- ```counts = logits.exp() # equivalent to the N matrix```
	- Normalize the counts; and they become probabilities
		- ```probs = counts / counts.sum(1, keepdim=True)```
5. The steps 4 (b &c) - together make a operation we call **SoftMax**.

6. Few notes: on **Smoothing**
	- In __bi-gram approach__, smoothing was done by adding a number (1) to the counts in N matrix.
		- Instead if we would have added a very large number (say 1,000,000), all counts would have become relatively equal.
		- And every bigram would have become equally likely.
		- Uniform distribution
	- __Gradient based framework__ - has an equivalent to smoothing
		- Suppose, if **W** was initialized as all 0.
			- All **logits** would have become 0.
				- All **counts** would have become 1.
					- **probs** would have turned out to be **uniform**

		- Trying to incentivizing **W** near 0;
			- Is equivalent to label smoothing
				- The more you incentivize that, more **smooth** distribution you'll achieve

7. Few notes: on **Regularization**(https://youtu.be/PaCmpygFfXo?t=6619)
	- We can augment the loss function
		- To have a small component called **regularization loss**

## Video 2
1. Constructing a sliding window of characters (should also be similar for tokens)
2. Indexing pytorch tensor with another 2d tensor