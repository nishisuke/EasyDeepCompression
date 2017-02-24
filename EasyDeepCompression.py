import caffe
import math
import numpy as np
from sklearn.cluster import KMeans
import struct

def initialize_network(deploy, phase, caffemodel=None):
    if caffemodel is None:
        if phase == 'test':
            return caffe.Net(deploy, caffe.TEST)
        else:
            return caffe.Net(deploy, caffe.TRAIN)

    if phase == 'test':
        return caffe.Net(deploy, caffemodel, caffe.TEST)
    else:
        return caffe.Net(deploy, caffemodel, caffe.TRAIN)

def extract(deploy, caffemodel, phase):
    dnn = initialize_network(deploy, phase, caffemodel)
    layers = dnn.params.keys()
    prefix = caffemodel.split('.')[0]
    np.save(prefix + '_layers.npy', layers)
    np.savez(prefix + '_weights.npz', **{l: dnn.params[l][0].data for l in layers})
    np.savez(prefix + '_biases.npz', **{l: dnn.params[l][1].data for l in layers})
    return prefix+'_layers.npy', prefix+'_weights.npz', prefix+'_biases.npz'
 
def prune_edges_with_small_weight(ndarray, percent):
    weights = ndarray.flatten()
    abso = np.absolute(weights)
    threshold = np.sort(abso)[int(math.ceil(weights.size * percent / 100))]
    weights[abso < threshold] = 0
    return weights.reshape(ndarray.shape)

def relative_index(absolute_index, ndarray, max_index):
    # absolute_index = [0, 1, 13, ...]
    # ndarrau =        [2, 1, 4, ...]
    first = absolute_index[0]
    relative = np.insert(np.diff(absolute_index), 0, first)
    dense = ndarray.tolist()
    max_index_or_less = relative.tolist()
    shift = 0
    for i in np.where(relative > max_index)[0].tolist():
        while max_index_or_less[i + shift] > max_index:
            max_index_or_less.insert(i + shift, max_index)
            dense.insert(i + shift, 0)
            shift += 1
            max_index_or_less[i + shift] -= max_index
    return (np.array(max_index_or_less), np.array(dense))
    # [0, 1, 7, 6, ...], [2, 1, 0, 4, ...]

def store_compressed_network(path, layers):
    with open(path, 'wb') as f:
        for layer in layers:
            f.write(struct.pack('Q', layer[1].size))
            f.write(layer[0].tobytes())
            f.write(layer[1].tobytes())
            f.write(struct.pack('Q', layer[2].size))
            f.write(layer[2].tobytes())
            f.write(layer[3].tobytes())
    return path

def compress(target_layers, weights, biases, pruning_percent, cluster_num, store_path_prefix):
    components = []

    for l in target_layers:
        # pruning
        sparse_1d = prune_edges_with_small_weight(weights[l], pruning_percent).flatten()
        # sparse_1d = [1, 2, 0, 0, 0, 0, 3]

        # K-means
        nonzero = sparse_1d[sparse_1d != 0]
        # nonzero = [1, 2, 3]
        clusters = KMeans(n_clusters=cluster_num).fit(nonzero.reshape(-1, 1))
        # if raise error, use this
        # clusters = KMeans(n_clusters=cluster_num).fit(nonzero.reshape(-1, 1).astype(np.float64))

        relative_index_in_4bits, cluster_labels = relative_index(np.where(sparse_1d != 0)[0], clusters.labels_ + 1, max_index = 16 - 1)
        if relative_index_in_4bits.size % 2 == 1:
            relative_index_in_4bits = np.append(relative_index_in_4bits, 0)
        pair_of_4bits_in_1byte = relative_index_in_4bits[np.arange(0, relative_index_in_4bits.size, 2)] * 16 + relative_index_in_4bits[np.arange(1, relative_index_in_4bits.size, 2)]

        components.append([pair_of_4bits_in_1byte.astype(np.dtype('u1')), cluster_labels.astype(np.dtype('u1')), clusters.cluster_centers_.astype(np.float32).flatten(), biases[l]])

    return store_compressed_network(store_path_prefix+'_'+str(pruning_percent)+'%_'+str(cluster_num)+'_clusters.npy', components)

def decode(deploy, compressed_network_path, phase, target_layers):
    caffemodel = initialize_network(deploy, phase)
    f = open(compressed_network_path, 'rb')
    for l in target_layers:
        edge_num = np.fromfile(f, dtype=np.int64, count=1)
        indices_pair_of_4bits = np.fromfile(f, dtype=np.dtype('u1'), count=int(math.ceil(edge_num/2.0)))
        cluster_labels = np.fromfile(f, dtype=np.dtype('u1'), count=edge_num)
        clusters_num = np.fromfile(f, dtype=np.int64, count=1)
        sharing_weights = np.fromfile(f, dtype=np.float32, count=clusters_num)

        biases = np.fromfile(f, dtype=np.float32, count=caffemodel.params[l][1].data.size)
        np.copyto(caffemodel.params[l][1].data, biases)

        relative = np.zeros(indices_pair_of_4bits.size*2, dtype=np.dtype('u1'))
        relative[np.arange(0, relative.size, 2)] = indices_pair_of_4bits / 16
        relative[np.arange(1, relative.size, 2)] = indices_pair_of_4bits % 16
        if relative[-1] == 0:
            relative = relative[:-1]

        weights = np.zeros(caffemodel.params[l][0].data.size, np.float32)
        index = np.cumsum(relative)
        weights[index] = np.insert(sharing_weights, 0, 0)[cluster_labels]
        np.copyto(caffemodel.params[l][0].data, weights.reshape(caffemodel.params[l][0].data.shape))
    remain = f.read()
    f.close()
    if len(remain) != 0:
        sys.exit("Decode error!")
    return caffemodel

def main(args):
    target_layers_path, weights_path, biases_path = extract(args.deploy, args.caffemodel, args.phase)
    target_layers = np.load(target_layers_path)
    # target_layers = ['conv1', 'conv2',...]
    weights = np.load(weights_path)
    # weights = {'conv1': [1, 2, ...]}
    biases = np.load(biases_path)
    # biases = {'conv1': [1, 2, ...]}
    compressed_network_path = compress(target_layers, weights, biases, float(args.pruning_percent), int(args.cluster_num), args.caffemodel.split('.')[0])
    compressed_caffemodel = decode(args.deploy, compressed_network_path, args.phase, target_layers)
    compressed_caffemodel_path = '.'.join(compressed_network_path.split('.')[:2]) + '.caffemodel'
    compressed_caffemodel.save(compressed_caffemodel_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('deploy')
    parser.add_argument('caffemodel')
    parser.add_argument('-p', '--phase', default='test')
    parser.add_argument('pruning_percent')
    parser.add_argument('cluster_num')
    args = parser.parse_args()

    main(args)
