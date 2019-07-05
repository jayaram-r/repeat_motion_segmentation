## Repeat Motion Segmentation
This repository contains implementation of a method for segmenting a time series with repeating patterns using dynamic time warping (DTW)-based sequence matching. It is assumed that the time series is captured from a sensor while a user is 
repeating one of a given set of actions multiple times. The actions can be in any order and repeated any number of times. For each action, the method takes as input a small number of template sequences that are representative 
of that action. For example, there could be ten action types and for each action there could be five template sequences. The search for DTW-based subsequence matching is made faster using techniques described in [1] and [2], where a 
series of less computationally complex lower bounds to the DTW distance are computed in order to prune out bad subseqeunce matches without actually calculating the DTW distance. Specifically, the method of [3] is used as a first-level 
crude lower bound, and the method of [4] is used as a second-level tighter lower bound for the DTW. Also, as suggested in [1], the template sequence and the subsequences are z-score normalized before calculating the distances.


### Dependencies
The list of Python dependencies can be found in the file `requirements.txt`. First, install the packages, preferably in an isolated enviroment (e.g. using [virtualenv](https://docs.python.org/3/tutorial/venv.html)), 
using the command:
```
pip install -r requirements.txt
```
In addition to the packages listed in `requirements.txt`, we use the package `DTAIDistance` [5], which has an efficient implementation of DTW in C (with Python bindings). This library also supports the usage of a warping window constraint 
(also known as Sakoe-Chiba constraint [6]) which can be used to speedup the DTW calculation by restricting the matching region in the 2D distance grid. Instructions to install this package can be found 
[here](https://dtaidistance.readthedocs.io/en/latest/usage/installation.html). It requires the `GCC` compiler suite and (optionally) the OpenMP API to be installed. Note that in order to install the C version, the package should be 
installed directly from the source (instead of using `pip`) as described in the installation page. For this reason, the package (`dtaidistance`) is not listed in `requirements.txt`. At the time of testing this code, version `1.2.2` 
of this package was used.


### Usage
To test the installation, run
```
python test_segmentation.py
```
which should generate a plot of the segmented time series in the file `segmentation.png`. A sample output is shown below:

![alt text](https://github.com/jayaram-r/repeat_motion_segmentation/blob/master/results/segmentation1.png "segmentation plot")

The function `segmentation.segment_repeat_sequences` can be used to segment an input time series. An example of this can be found in `test_segmentation.py` and is briefly explained below.
```python
data_segments, labels = segment_repeat_sequences(data_sequence, template_sequences, normalize=True, 
                                                 normalization_type='z-score', warping_window=0.5, alpha=0.75)                                    
```
#### Inputs
- `data_sequence`: numpy array of shape `(n, 1)` with the values of the time series to be segmented. For example,
```
>>> import numpy as np
>>> data_sequence = np.random.rand(10, 1)
```
- `template_sequences`: list `[L_1, . . ., L_k]`, where each `L_i` is another list `L_i = [s_i1, . . ., s_im]`, and each `s_ij` is a numpy array (of shape (m, 1)) corresponding to a template sequence. For example, the example below has 
three actions with five template sequences per action.
```
>>> L1 = [np.random.rand(10, 1) for i in range(5)]
>>> L2 = [np.random.rand(12, 1) for i in range(5)]
>>> L3 = [np.random.rand(15, 1) for i in range(5)]
>>> template_sequences = [L1, L2, L3]
```
- `normalize`: Set to `True` in order to normalize both the data and template sequences. Recommended to set this to `True`.
- `normalization_type`: Set to either `'z-score'` or `'max-min'` which specifies the type of normalization. Typically, z-score normalization is recommended.
- `warping_window`: Size of the warping window used to constrain the DTW matching path. This is also know as the Sakoe-Chiba band in DTW literature [6]. This can be set to `None` if no warping window
                    constraint is to be applied; else it should be set to a fractional value in `(0, 1]`. The actual warping window is obtained by multiplying this fraction with the length of the
                    longer sequence. Suppose this window value is `w`, then any point `(i, j)` along the DTW path satisfies `|i - j| <= w`. For example, if `warping_window = 0.5` and the length of the longer sequence is 100, then the 
                    warping window constraint is `|i - j| <= 50`. Setting this to a large value (closer to 1), allows the warping path to be flexible, while setting it to a small value (closer to 0) will constrain the warping path 
                    to be closer to the diagonal. Note that a small value can also speed-up the DTW calculation significantly.
- `alpha`: float value in the range `(0, 1)`, but recommended to be in the range `[0.5, 0.8]`. This value controls the search range for the subsequence length. If `m` is the median length of the template sequences, then the search 
           range for the subsequences is obtained by uniform sampling of the interval `[alpha * m, (1 / alpha) * m]`. A smaller value of `alpha` increases the search interval of the subsequence length resulting in a higher search 
           time, but also a more extensive search for the best match. On the other hand, a larger value of `alpha` (e.g. 0.85) will result in a faster but less extensive search.
           
#### Outputs
- `data_segments`: list of numpy arrays, where each array corresponds to a segment of the input sequence `data_sequence`. In other words, `data_segments = [seg_1, seg_2, . . ., seg_k]`, where `seg_i` is a numpy array of shape `(n_i, 1)` 
                   corresponding to the `i-th` segment.
- `labels`: list of best-matching template labels for each of the subsequences in `data_segments`. The labels take values in `{1, 2, . . ., m}`, where `m` is the number of actions, and label `j` at index `i` means that the subsequence
            `data_segments[i]` is mapped to the templates in position `j - 1` of the input list `template_sequences`.


### References
1. Rakthanmanon, Thanawin, et al. "Searching and mining trillions of time series subsequences under dynamic time warping." Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2012.
1. Mueen, Abdullah, and Eamonn Keogh. "Extracting optimal performance from dynamic time warping." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016. [slides](https://www.cs.unm.edu/~mueen/DTW.pdf).
1. Kim, Sang-Wook, Sanghyun Park, and Wesley W. Chu. "An index-based approach for similarity search supporting time warping in large sequence databases." Proceedings 17th International Conference on Data Engineering. IEEE, 2001.
1. Yi, Byoung-Kee, H. V. Jagadish, and Christos Faloutsos. "Efficient retrieval of similar time sequences under time warping." Proceedings 14th International Conference on Data Engineering. IEEE, 1998.
1. https://dtaidistance.readthedocs.io/en/latest/index.html
1. Sakoe, Hiroaki, et al. "Dynamic programming algorithm optimization for spoken word recognition." Readings in speech recognition 159 (1990): 224.

  
