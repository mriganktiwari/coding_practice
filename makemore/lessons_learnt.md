# Lessons learnt from Karpathy videos:
---

## Video 1:
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
	- equivalent to __maximum conditional probability__ - of sequence of chars occuring together (0-1).
	- equivalent to maximizing the __log likelihood__ (because log is monotonic) (-inf to 0).
	- equivalent to minimizing the __negative log likelihood__ (0 to inf).
	- equivalent to minimizing the __average negative log likelihood__.
11. Model smoothing:
	- In bi-gram model: pairs that never occured together in a dataset, will have their count = 0 in count matrix. This would introduce ```-inf``` in ```logprob```.
		- To handle such instances, 1 is added to all counts in the matrix
	- 