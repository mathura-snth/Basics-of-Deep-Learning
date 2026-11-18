# -*- coding: utf-8 -*-
"""
SOM hierarchique probabiliste

@author: Guenael Cabanes
"""

from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn import datasets    
import numpy as np
import matplotlib.pyplot as plt


class PrSOM:
    def __init__(self, data, shape):
        self.shape = shape
        self.N, self.dim = data.shape
        self.data = data
     
        # Initialisation des prototypes selon les axes de l'ACP
        self.pca = PCA().fit(self.data) 
        var = self.pca.transform(data).std(axis = 0)[0:2]
        rangex = np.arange(-var[0],var[0],var[0]*2/self.shape[0])
        if len(rangex)>self.shape[0]:
            rangex = rangex[:-1]
        rangey = np.arange(-var[1],var[1],var[1]*2/self.shape[1])
        if len(rangey)>self.shape[1]:
            rangey = rangey[:-1]  
        self.W = []
        for x in rangex:
            for y in rangey:
                self.W.append([x,y])
        self.W = np.hstack((np.array(self.W), np.zeros((len(self.W),self.dim-2))))
        self.W =self.pca.inverse_transform(self.W[:,:self.pca.components_.shape[0]])
        if np.sum(self.data) == self.N:
            self.W[self.W<0]=0
        
        # Initialization des temperatures (radius de voisinage)
        self.s0 = max(1,max(self.shape)/4)
        self.sf = 1
        self.s = self.s0

        # Initialization de beta
        d = self.dist(self.W,self.W)
        d[d==0] = 'NaN'
        self.beta = 1/(np.max(np.nanmin(d,axis=0))**2)

        # Nombre de neurones
        self.nb_neuron = int(np.product(self.shape))        
        
        # Coordonnée des neurones
        self.coord = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.coord.append([(i,j)])
        
        # Distance euclidienne entre les prototypes de la première couche
        self.distmat = np.zeros((self.nb_neuron, self.nb_neuron))      
        for n in range(self.nb_neuron):
            for m in range(self.nb_neuron):
                self.distmat[n,m] = euclidean(self.coord[n],self.coord[m])

        # Initialization de la matrice P de probabilité entre chaque donnée et chaque neurone
        self.P = np.zeros((self.N, self.nb_neuron))+1/self.nb_neuron
        
        # Initialization de la matrice q de probabilité à priori pour chaque neurone
        self.q = np.zeros(self.nb_neuron)+1/self.nb_neuron
        
     
    def Kij(self, T): # Calcul de la fonction de voisinage
        self.s = self.s0*(self.sf/self.s0)**(T/self.Tmax)
        
        if self.s>0:
            return np.exp(-self.distmat**2/(2*self.s**2))
        else:
            return np.identity(self.nb_neuron)
        
        
    def train(self,Tmax): #Apprentissage
        self.Tmax = Tmax
        
        for T in range(Tmax):
            # Calcul de K, la fonction de voisinage
            self.K = self.Kij(T)
            
            # Affectation des données aux neurones (calcul de P)
            self.assign(self.data)
            
            # Mise à jours de prototypes (calcul de W)
            self.update()

    
    def dist(self, X1, X2):  # Calcul des distances 
        if np.sum(self.data) == self.N:
            return 1/np.sqrt(2)*euclidean_distances(np.sqrt(X1), np.sqrt(X2)) ## Hellinger pour probabilités
        else:
            return euclidean_distances(X1, X2)  ## Euclidienne pour vecteurs (première couche)
            
        
    def assign(self, X):  ## Calcul de la matrice P de probabilité entre chaque donnée et chaque neurone 
        # Distance entre les données et les prototypes
        self.Dxw = 1/2*self.dist(X, self.W)**2
        
        # Création de la matrice Q
        Q = np.tile(self.q,(self.N,1)).T 
        
        # Calcul de P
        self.P = Q * np.dot(self.K,np.exp(-self.beta*self.Dxw.T)) 
        norm = np.tile(np.sum(self.P, axis=0),(self.nb_neuron,1))+1e-16
        self.P = self.P / norm       
        
        return self.P


    def update(self): ## calcul de W, la matrice des prototypes
        # Calcul de P * K
        KP = np.dot(self.K,self.P)
    
        #calcul de W
        norm = np.tile(np.sum(KP, axis=1),(self.dim,1)).T
        self.W = np.dot(KP,self.data) / norm
    
    
    def Poutput(self): # Calcul et mise en forme de la sortie des couches
    
        mask = self.shape
        P = self.assign(self.data).T
        self.out = []

        for p in P:
            pmat = p.reshape(mask)                            
            self.out.append(pmat.reshape((1,mask[0]*mask[1]))[0])
            
        self.out = np.array(self.out)
        
        return (self.out)
                    
            
class DeepSOM:
    def __init__(self, data, N_layers, shape):
        self.data = data
        self.N_layers = N_layers
        self.shape = shape
        self.out = []    # initialisation de la sortie
    
    def train(self, Tmax):
        X = self.data
        self.layers = [] #liste des couches
        
        # pour chaque couche :
        for i in range(self.N_layers): 
            
            # Met à jours la taille de la carte 
            self.shape = (np.max((self.shape[0]-2,4)), np.max((self.shape[1]-2,4)))
            
            # Initialisation de la couche
            l = PrSOM(X, shape = self.shape)
                 
            # Apprentissage de la couche
            l.train(Tmax)
                
            # Calcul de la sortie de la couche
            X = l.Poutput()
            
            # Mise en mémoire de la couche
            self.layers.append(l)
        
        self.out = X
        
    def scores(self):  # Affichage des scores Ijs sur les differentes couches
        l=0
        for lay in self.layers:
            l+=1
            d = euclidean_distances(lay.out)
            ma = np.max(d,axis=0)
            d[d==0] = 1
            mi = np.min(d,axis=0)
            index = np.mean(mi/ma)
            print("Couche ",l,": score = ", round(index,3))
 

             
 

# Tests sur données réelles

T=50
N_layers = 5
shape = (5,5)

data = datasets.load_iris()
X = data.data
Y = data.target

S = DeepSOM(X, N_layers, shape)
S.train(T)
S.scores()

plt.imshow(S.out, interpolation = "nearest", aspect ="auto")



























# from sklearn import datasets
# iris = datasets.load_iris()
# X = iris.data
# Y = iris.target
# S = PrSOM(X, shape = (10,10), beta = 'auto')
# S.visu()
# S.train(100, vis=False)
# S.visu()

# X2 = S.Poutput((10,10))[0]
# S2 = PrSOM(X2, shape = (8,8), beta = 'auto')
# S2.visu()
# S2.train(100, vis=False)
# S2.visu()

# X2 = S2.Poutput((8,8))[0]
# S2 = PrSOM(X2, shape = (5,5), beta = 'auto')
# S2.visu()
# S2.train(100, vis=False)
# S2.visu()


#class Neuron:
#    def __init__(self, dim, nb_neuron, coord):
#        self.w = rand(dim)
#        self.q = 1/nb_neuron
#        self.coord = coord
#        
#    def Dpx(self, x):
#        return euclidean(self.w,x)
