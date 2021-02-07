import numpy as np
import math
import scipy.sparse.linalg as sla

def DataProjection(X,r,type='NormalProj'):

    
    if r == 0:
        Xp = X
    else:
    
        D = X.shape[0]

        if type=='PCA':
            mX=X.mean(axis=1).reshape(D,1)
            X=X-mX
            C=np.cov(X)
            [V, U]= np.linalg.eigh(C)
            U=np.flip(U,axis=1)[:r]
            Xp = U.T @ X
            ProjMat=U.T
        
        elif type == 'NormalProj':

            npr = np.random.normal(0, 1/math.sqrt(r), [r*D,1])
            PrN =  npr.reshape(r,D).T
            Xp = PrN @ X
            ProjMat=PrN

        
        elif type == 'BernoulliProj':

            bp = np.random.random([r*D,1])
            Bp = 1/math.sqrt(r) * (bp >= .5) - 1/math.sqrt(r) * (bp < .5)
            PrB = Bp.reshape(r,D).T
            Xp = PrB @ X
            ProjMat=PrB

        #Achlioptas        
        elif type == 'Sparse':
            pass





def DiSC(W,dim,final_clusters,l=0.01,Ensembles=50):

    
    lambda_0=l
    
    K=min(final_clusters*dim,W.shape[0]) 

    dim=min(K,dim) #the intrinsic dimenionsality of the subspaces

    N=W.shape[1] #The total number of points


    PPCluster=dim+3 #points per cluster
    Clusters=max(10,math.floor(N*0.1)) #number of clusters

    
    Is=np.ones([N,Ensembles])*-1
    Weights=np.zeros([1,Ensembles])
    l=np.ones([1,Ensembles])*l
    I_CL=np.zeros([N,Ensembles])
    ProjectionMatrix=np.zeros([Ensembles,1])


    for iter in range(Ensembles):
        #do a new random projection at every iteration
        [X,ProjMat] = DataProjection(W,K,'NormalProj')


from numpy import genfromtxt
W = genfromtxt('subspace_data.csv',delimiter=',').T #Data matrix dims x points


DiSC(W,1,2,0.1,100)
