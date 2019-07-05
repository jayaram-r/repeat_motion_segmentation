## Repeat Motion Segmentation
This repo contains an implementation of a method for segmenting a time series with repeating patterns using dynamic time warping (DTW)-based sequence matching. It is assumed that the time series is captured from a sensor while a user is 
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
installed directly from the source (instead of using `pip`) as described in the installation page [5]. For this reason, the package (`dtaidistance`) is not listed in `requirements.txt`. At the time of testing this code, version `1.2.2` 
of this package was used.

### Usage


### References
1. Rakthanmanon, Thanawin, et al. "Searching and mining trillions of time series subsequences under dynamic time warping." Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2012.
1. Mueen, Abdullah, and Eamonn Keogh. "Extracting optimal performance from dynamic time warping." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016. [slides](https://www.cs.unm.edu/~mueen/DTW.pdf).
1. Kim, Sang-Wook, Sanghyun Park, and Wesley W. Chu. "An index-based approach for similarity search supporting time warping in large sequence databases." Proceedings 17th International Conference on Data Engineering. IEEE, 2001.
1. Yi, Byoung-Kee, H. V. Jagadish, and Christos Faloutsos. "Efficient retrieval of similar time sequences under time warping." Proceedings 14th International Conference on Data Engineering. IEEE, 1998.
1. https://dtaidistance.readthedocs.io/en/latest/index.html
1. Sakoe, Hiroaki, et al. "Dynamic programming algorithm optimization for spoken word recognition." Readings in speech recognition 159 (1990): 224.

  
