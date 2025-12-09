from sklearn.cluster import KMeans
import numpy as onp

def make_snowstorm_anchors_nd(alpha_all, n_anchors=15):
    alpha_all = onp.asarray(alpha_all)
    km = KMeans(n_clusters=n_anchors, n_init=10).fit(alpha_all)
    alpha_points = km.cluster_centers_
    
    # yields = number of events in each cluster
    labels = km.labels_
    yields = onp.bincount(labels).astype(float)

    return alpha_points, yields