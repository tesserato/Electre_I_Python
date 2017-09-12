#%%
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision = 2)

def readtxt(path , separator = ',' , decimal = '.'):
    F = open(path, 'r' , encoding='utf8')
    data = []
    data_dict = {}
    for line in F:
        if line != '\n':
            data.append(line.replace(decimal, '.').replace('\n', '').split(separator))
    F.close()
    header = data.pop(0)
    for line in data:
        data_dict[line[0]] = [float(cell) for cell in line[1:]]

    # data = np.matrix(data, dtype='float32')
    # print(data_dict)
    return data_dict, header

def normalize(data, amin = 0, amax = 1):
    max_per_col = [max(col) for col in np.matrix([row for row in data.values()]).T.tolist()]
    min_per_col = [min(col) for col in np.matrix([row for row in data.values()]).T.tolist()]
    # print(max_per_col)
    # print(min_per_col)
    # exit()    
    normalized_data = {}
    for key in data:
        normalized_row = []
        for i, cell in enumerate(data[key]):
            normalized_row.append( amin + (amax - amin) * (cell - min_per_col[i]) / (max_per_col[i] - min_per_col[i]) )
        normalized_data[key] = normalized_row
    return normalized_data



countries, header = readtxt('countries.csv')

# create initial 'number_of_clusters' random clusters
clusters = []
number_of_clusters = 6
for n in range(number_of_clusters):
    cluster = {}
    cluster['centroid'] = random.choice([v for v in countries.values()])
    cluster['components'] = []
    clusters.append(cluster)

last_average_distance = 0

def step(iter):
    global last_average_distance
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    initial_clusters = clusters
    #assign countries to the closest cluster
    average_distance = []
    for country in countries:        
        distances = []
        pos = np.array(countries[country])
        for cluster in clusters:
            cent = np.array(cluster['centroid'])
            distances.append(np.sqrt(np.sum(np.power(cent - pos,2))))
        clusters[distances.index(min(distances))]['components'].append(country)
        average_distance.append(min(distances))
    average_distance = np.mean(average_distance)
    #graph and recalculate centroid
    i = 0
    colors = ['r','b','m','k','g','c','y','k','w']
    for cluster in clusters:
        X = []
        Y = []
        Z = []    
        for country in cluster['components']:
            x = countries[country][0]
            X.append(x)
            y = countries[country][1]
            Y.append(y)
            z = countries[country][2]
            Z.append(z)
            ax.scatter(x, y, z , c=colors[i])
            ax.text(x, y, z, country)
            ax.plot([cluster['centroid'][0], x], [cluster['centroid'][1], y], [cluster['centroid'][2], z], color =colors[i])        
        i += 1
        cluster['centroid'][0] = np.mean(X)
        print(cluster['centroid'], np.mean(X))
        cluster['centroid'][1] = np.mean(Y)
        print(cluster['centroid'])
        cluster['centroid'][2] = np.mean(Z)
        print(cluster['centroid'])
        cluster['components'] = [] #erase components to be recomputed in next step

    plt.title('Distância média = ' + str(average_distance))
    ax.set_xlabel('X: atributo na primeira coluna')
    ax.set_ylabel('Y: atributo na segunda coluna')
    ax.set_zlabel('Z: atributo na terceira coluna')
    # plt.show()
    fig.savefig(str(iter) + ' iteração com ' + str(number_of_clusters) + ' clusters.png')
    # fig.clf()
    if last_average_distance == average_distance:
        exit()
    last_average_distance = average_distance
    return initial_clusters

for ctr in range(20):
    step(ctr)
    # print(clusters)




