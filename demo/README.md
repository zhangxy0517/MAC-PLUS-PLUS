# Usage:
1. Ensure this folder and the executable files (`Boot`, `MAC`) are in the same folder.
2. Run the following command.
```
./Boot --demo
```
The output will be like:
```
 graph construction: 1.9585
 coefficient computation: 0.0348983
0.180577->min(0.278128 0.180577 0.66339)
 inliers: 85    total num: 6614 inlier ratio: 1.28515%
 clique computation: 0.018112
2077 sampled correspondences have 68 inlies: 3.27395%
 clique selection: 0.0116206
too many cliques
Total 6614 cliques form 851 clusters.
Mat 0 in cluster 0 (84) is similar to best_est1 with score 1.27367e-07
Both choose cluster 0
best_est1: 0.617764, best_est2: 0.601344
prior is better
60 intersection correspondences have 52 inlies: 86.6667%
Center est: 1, RE = 3.35649, TE = 1.99945, score = 19.8047
 post evaluation: 0.24024
0.954023 0.976471 0.965116
RE=3.35649 TE=1.99945
 0.805857   0.14761 -0.573416 -0.980854
-0.271891  0.952541 -0.136899 -0.785434
 0.525995  0.266227  0.807746   1.10564
        0         0         0         1
```
3. Run the Python script `vis.py`. You will see the visualizations.
![image](https://github.com/user-attachments/assets/e69bffb6-86e6-4683-8816-719b833e51ca)
![image](https://github.com/user-attachments/assets/ac19990e-af88-4499-aea5-a7f112abfbb8)
![image](https://github.com/user-attachments/assets/334da21c-e10a-4fad-9b92-586c63e5998b)

4. Run the Python script `cluster_vis.py` to visualize the transformation clustering result (You must sign up for the chart studio at plotly.com).

![image](https://github.com/user-attachments/assets/4998e861-9c99-46d4-98f3-d6ac79244a09)
