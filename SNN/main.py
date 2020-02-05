import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabaz_score
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from snn import SNN

# Note: the code in this file was built taking as base code a sci-kit learn example on synthetic data set generation and plotting the results of different clustering algorithms: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html


def main():

    """Main function"""

    # seed for reproducibility of results
    np.random.seed(0)

    # simple synthetic data
    n_samples = 1500
    circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.1)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # anisotropic distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5],  random_state=random_state)

    # blobs with very different densities
    diff_density_blobs = datasets.make_blobs(n_samples=[n_samples//2, n_samples//5, n_samples//4, n_samples//6, n_samples//5, n_samples//3], cluster_std=[.1, .5, .2, .3, 2, 6], random_state=6)
    blobs0 = datasets.make_blobs(n_samples=n_samples, cluster_std=1., random_state=1)
    blobs1 = datasets.make_blobs(n_samples=n_samples, cluster_std=0.2, random_state=2)
    blobs2 = datasets.make_blobs(n_samples=n_samples//3, cluster_std=12, random_state=1)
    diff_density_blobs1 = (np.concatenate((blobs0[0], blobs1[0], blobs2[0])), np.concatenate((blobs0[1], blobs1[1]+3, blobs2[1]+3)))

    # real datasets
    iris = datasets.load_iris(return_X_y=True)
    faces = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4, return_X_y=True)
    breast = datasets.load_breast_cancer(return_X_y=True)

    default_base = {'eps': .3,
                    'n_neighbors': 20,
                    'min_shared_neighbor_proportion': 0.5,
                    'n_clusters': 3,
                    'plot_index_x': 0,
                    'plot_index_y': 1,
                    'plot_half_width': 2.5,
                    'plot_half_height': 2.5}

    for i, tested_datasets in enumerate([
        [('Circles', circles, {'n_clusters': 2}),
        ('Circles (noisy)', noisy_circles, {'eps': .15, 'n_clusters': 2}),
        ('Moons', noisy_moons, {'n_clusters': 2}),
        ('Varied', varied, {'eps': .18}),
        ('Ansiotropic', aniso, {'eps': .15}),
        ('Blobs', blobs, {}),
        ('Square', no_structure, {})],

        [('Different density', diff_density_blobs, {'eps': .15, 'n_clusters': 6}),
        ('Different density (II)', diff_density_blobs1, {'eps': .15, 'n_clusters': 9})],

        [("Iris", iris, {'eps': .8, 'plot_index_x': 2, 'plot_index_y': 1, 'plot_half_width': 3, 'plot_half_height': 2.5}),
        ('Breast cancer', breast, {'plot_index_x': 2, 'eps': 2, 'plot_index_y': 3, 'n_clusters': 2, 'n_neighbors': 55, 'min_shared_neighbor_proportion': 0.5}),
        ('Faces', faces, {'plot_index_x': 10, 'plot_index_y': 3, 'eps': 25, 'n_clusters': 5749//10, 'n_neighbors': 10, 'min_shared_neighbor_proportion': 0.5})]
    ]):

        plt.figure(num='Comparison of clustering algorithms ({})'.format(i+1), figsize=(40, 30))
        plot_num = 1

        for i_dataset, (data_name, dataset, algo_params) in enumerate(tested_datasets):
            # update parameters with dataset-specific values
            params = default_base.copy()
            params.update(algo_params)

            X, y = dataset

            # normalize dataset for easier parameter selection
            X = StandardScaler().fit_transform(X)

            # algorithms
            k_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
            spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity="nearest_neighbors")
            dbscan = cluster.DBSCAN(eps=params['eps'])
            snn = SNN(neighbor_num=params['n_neighbors'], min_shared_neighbor_proportion=params['min_shared_neighbor_proportion'])

            clustering_algorithms = (
                ('K-Means', k_means),
                ('Spectral', spectral),
                ('DBSCAN', dbscan),
                ('SNN', snn)
            )

            warnings.simplefilter("ignore")

            for name, algorithm in clustering_algorithms:

                algorithm.fit(X)

                if hasattr(algorithm, 'labels_'):
                    y_pred = algorithm.labels_.astype(np.int)
                else:
                    y_pred = algorithm.predict(X)

                # evaluate the results
                mutual_info = None
                rand_index = None
                calinski_score = None
                if len(np.unique(y_pred)) > 1 and len(np.unique(y)) > 1:
                    mutual_info = adjusted_mutual_info_score(y, y_pred, average_method='arithmetic')
                    rand_index = adjusted_rand_score(y, y_pred)
                    calinski_score = calinski_harabaz_score(X, y_pred)

                plt.subplot(len(tested_datasets), len(clustering_algorithms), plot_num)
                if i_dataset == 0:
                    plt.title(name, size=18)

                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                     '#f781bf', '#a65628', '#984ea3',
                                                     '#999999', '#e41a1c', '#dede00']),
                                              int(max(y_pred) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(X[:, params['plot_index_x']], X[:, params['plot_index_y']], s=10, color=colors[y_pred])
                plt.xlim(-params['plot_half_width'], params['plot_half_width'])
                plt.ylim(-params['plot_half_height'], params['plot_half_height'])
                plt.xticks(())
                plt.yticks(())
                if mutual_info and rand_index and calinski_score:
                    plt.text(.99, .01, ('MI=%.2f RI=%.2f CHS=%.2f' % (mutual_info, rand_index, calinski_score)), transform=plt.gca().transAxes, size=12, horizontalalignment='right')
                plot_num += 1

        plt.show(block=False)
        plt.pause(0.05)

    input("Press Enter to exit.")


if __name__ == "__main__":
    main()
