# LAB 5         : Profiling
# Author        : Raghavendra Belapure, Karanjit Singh Cheema
# GTID          : 902866342, 902757702
-------------------------------------------------------------------------------
# Questions for Lab 5

## Part 0: Getting started

* *What is the name of the processor which you used for this assignment?*
Answer : Intel(R) Xeon(R) CPU X5650  @ 2.67GHz

Part 1: Profiling
---------------------

Note : Every average reading below is obtained from running the program 10 
       times and taking the average of all the 10 readings.

* *What is the IPC of the utility in our use case? Is it good?*
Answer : The average IPC is 1.27
	 On jinx cluster, maximum possible IPC is 4 (in rare cases 5). IPC of 
	 2 is considered as a good IPC. Thus, it can be said that this IPC is
	 not that good.

* *What is the fraction of mispredicted branches? Is it acceptable?*
Answer : The average branch misprediction rate is 0.39%. As this number is much
	 less than 5%, we can say that this is completely acceptable. In fact,
	 the branch prediction accuracy for this program is very good.

* *What is the rate of cache misses? How do you feel about this number?*
Answer : On an average, 25.749% of the time cache-misses are incurred during 
	 execution of this program. As the cache miss rate is more than 10%,
	 this is a very high cache miss rate.


* *Which two functions take most of the execution time? What do they do?*
Answer : The two functions that take most of the execution time are -
	 1. MorphologyApply (52.24%) - This performs the 2-D transformations
	    on the image.
	 2. WriteOnePNGImage (35.49%) - This function outputs the transformed
	    image and writes it on disk.

Part 2: Compiler Optimizations
------------------------------

* *What is the "User Time" for program execution before you start optimizing?*
Answer : The average "User Time" before optimization is 5.452 sec.

* *What is the "User Time" for program execution after you completed **all** three steps and rune the program with `-fprofile-use`*?
Answer : The average "User Time" after profile guided optimization is 7.106 sec.
