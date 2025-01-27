*Exercise: Cluster Analysis: To better understand what typical charging sessions look like, carry out a cluster
 analysis to provide management with a succinct report of archetypical charging events. Think of an
 appropriate trade-off between explainability and information content and try to come up with names
 for these clusters. What is the value of identifying different types of charging sessions?*


**Data Preparation and Analysis Approach**:

1.**Initial Decision and Feature Selection**: 
We decided to begin our analysis using the KMeans clustering algorithm. 
Initially, we focused on two features for clustering: `duration (charging time)` and `kWhDelivered (energy delivered)`, 
as these seemed to capture the key aspects of each session. While we also considered adding `chargingPower` for a 3D analysis, 
the resulting plot was not as informative, so we opted to stick with the two original features for simplicity and clarity.

2.**Choosing the Number of Clusters**: 
To determine the optimal number of clusters, we applied the Elbow Method.  
By visualizing the sum of squared errors against the number of clusters, we were able to identify the `elbow` at around 3 to 4.

3.**Cluster Analysis**:
We proceeded with the KMeans algorithm to assign each session to one of the clusters. 
The clusters were defined based on the similarity of charging times and energy delivered, 
which allowed us to identify distinct patterns in the data.

4.**Evaluation of Cluster Quality**: 
After clustering, we performed a Silhouette Analysis to assess the quality of our clusters.
The silhouette score of 0.48 suggests that there is room for improvement. 
In the next steps, we plan to experiment with different clustering algorithms, 
such as DBSCAN or hierarchical clustering, and potentially refining the feature selection or scaling methods.

