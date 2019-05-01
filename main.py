import matplotlib.pyplot as plt
import SynteticData as sd
import KMeansmp as kmmp
import numpy


generator = sd.SynteticData()
X = generator.generate_data(clusters=20,num_points=4000)
plt.plot(X[:,0], X[:,1],'ro')
plt.show()

kmeans = kmmp.KMeansmp(n_clusters=20, n_init=3, max_iter=200)
kmeans = kmeans.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_.astype(float), s=20)
plt.scatter(kmeans.centroids_[:,0],kmeans.centroids_[:,1],c='r',s=80)
plt.show()
