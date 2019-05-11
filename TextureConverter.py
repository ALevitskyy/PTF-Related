#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:21:21 2019

@author: andriylevitskyy
"""
import sys
sys.path.append("/usr/local/lib/python3.7/site-packages")
import numpy as np
from scipy.io  import loadmat
import scipy.spatial.distance
import pickle
import pandas as pd

def get_blender():
    # Only required for precomputing lookup matrix
    vertex_map = pd.DataFrame(pickle.load(open("vertex_map.pkl","rb")))
    new_map = []
    for i in np.unique(vertex_map["faceidx"]):
        mini_map = vertex_map[vertex_map["faceidx"]==i]
        face = {"verteces":np.array(mini_map["vertidx"]),
                "uvs":np.vstack([mini_map["u"],mini_map["v"]])}
        new_map.append(face)
    new_map = pd.DataFrame(new_map)
    verteces = []
    uvs = []
    for i in new_map["verteces"]:
        verteces.append(i)
    for i in new_map["uvs"]:
        uvs.append(i)
    verteces = np.array(verteces)
    uvs = np.array(uvs)
    return verteces,uvs

verteces,uvs = get_blender()

def get_uvs(vert_indices):
    # Only required for precomputing lookup matrix
    index = np.where(np.all(verteces==np.array(vert_indices),axis=1))
    return uvs[index].squeeze()

precomputed = pickle.load(open("precomputed.pkl","rb"))

class DensePoseMethods:
    def __init__(self):
        #
        ALP_UV = loadmat("UV_Processed.mat")
        self.FaceIndices = np.array( ALP_UV['All_FaceIndices']).squeeze()
        self.FacesDensePose = ALP_UV['All_Faces']-1
        self.U_norm = ALP_UV['All_U_norm'].squeeze()
        self.V_norm = ALP_UV['All_V_norm'].squeeze()
        self.U = ALP_UV['All_U'].squeeze()
        self.V = ALP_UV['All_V'].squeeze()
        self.All_vertices =  ALP_UV['All_vertices'][0]
        ## Info to compute symmetries.
        self.SemanticMaskSymmetries = [0,1,3,2,5,4,7,6,9,8,11,10,13,12,14]
        self.Index_Symmetry_List = [1,2,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17,20,19,22,21,24,23]    
        self.vertex_map = pd.DataFrame(pickle.load(open("vertex_map.pkl","rb")))
    
    def barycentric_coordinates_exists(self,P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        #
        vCrossW = np.cross(v,w)
        vCrossU = np.cross(v, u)
        if (np.dot(vCrossW, vCrossU) < 0):
            return False;
        #
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        #
        if (np.dot(uCrossW, uCrossV) < 0):
            return False;
        #
        denom = np.sqrt((uCrossV**2).sum())
        r = np.sqrt((vCrossW**2).sum())/denom
        t = np.sqrt((uCrossW**2).sum())/denom
        #
        return((r <=1) & (t <= 1) & (r + t <= 1))

    def barycentric_coordinates(self,P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        #
        vCrossW = np.cross(v,w)
        vCrossU = np.cross(v, u)
        #
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        #
        denom = np.sqrt((uCrossV**2).sum())
        r = np.sqrt((vCrossW**2).sum())/denom
        t = np.sqrt((uCrossW**2).sum())/denom
        #
        return(1-(r+t),r,t)

    def IUV2FBC( self, I_point , U_point, V_point):
        P = [ U_point , V_point , 0 ]
        FaceIndicesNow  = np.where( self.FaceIndices == I_point )
        FacesNow = self.FacesDensePose[FaceIndicesNow]
        #
        P_0 = np.vstack( (self.U_norm[FacesNow][:,0], self.V_norm[FacesNow][:,0], np.zeros(self.U_norm[FacesNow][:,0].shape))).transpose()
        P_1 = np.vstack( (self.U_norm[FacesNow][:,1], self.V_norm[FacesNow][:,1], np.zeros(self.U_norm[FacesNow][:,1].shape))).transpose()
        P_2 = np.vstack( (self.U_norm[FacesNow][:,2], self.V_norm[FacesNow][:,2], np.zeros(self.U_norm[FacesNow][:,2].shape))).transpose()
        #

        for i, [P0,P1,P2] in enumerate( zip(P_0,P_1,P_2)) :
            if(self.barycentric_coordinates_exists(P0, P1, P2, P)):
                [bc1,bc2,bc3] = self.barycentric_coordinates(P0, P1, P2, P)
                return(FaceIndicesNow[0][i],bc1,bc2,bc3)
        #
        # If the found UV is not inside any faces, select the vertex that is closest!
        #
        D1 = scipy.spatial.distance.cdist( np.array( [U_point,V_point])[np.newaxis,:] , P_0[:,0:2]).squeeze()
        D2 = scipy.spatial.distance.cdist( np.array( [U_point,V_point])[np.newaxis,:] , P_1[:,0:2]).squeeze()
        D3 = scipy.spatial.distance.cdist( np.array( [U_point,V_point])[np.newaxis,:] , P_2[:,0:2]).squeeze()
        #
        minD1 = D1.min()
        minD2 = D2.min()
        minD3 = D3.min()
        #
        if((minD1< minD2) & (minD1< minD3)):
            return(  FaceIndicesNow[0][np.argmin(D1)] , 1.,0.,0. )
        elif((minD2< minD1) & (minD2< minD3)):
            return(  FaceIndicesNow[0][np.argmin(D2)] , 0.,1.,0. )
        else:
            return(  FaceIndicesNow[0][np.argmin(D3)] , 0.,0.,1. )
        
        
    def barycentric_coordinates_fast(self, P0, P1, P2, P):
        # This is a merge of barycentric_coordinates_exists & barycentric_coordinates.
        # Inputs are (n, 3) shaped.

        u = P1 - P0   #u is (n,3)
        v = P2 - P0   #v is (n,3)
        w = P.T - P0    #w is (n,3)
        #
        vCrossW = np.cross(v, w) #result is (n,3)
        vCrossU = np.cross(v, u) #result is (n,3)
        A = np.einsum('nd,nd->n', vCrossW, vCrossU) # vector-wise dot product. Result shape is (n,)
        #
        uCrossW = np.cross(u, w)
        uCrossV = - vCrossU
        B = np.einsum('nd,nd->n', uCrossW, uCrossV)
        #
        sq_denoms = np.einsum('nd,nd->n', uCrossV, uCrossV) #result shape is  (n,)
        sq_rs = np.einsum('nd,nd->n', vCrossW, vCrossW)
        sq_ts = np.einsum('nd,nd->n', uCrossW, uCrossW)
        rs = np.sqrt(sq_rs / sq_denoms)  #result shape is  (n,)
        ts = np.sqrt(sq_ts / sq_denoms)
        #
        results = [None] * P0.shape[0]
        for i in range(len(results)):
            if not (A[i] < 0 or B[i] < 0):
                if ((rs[i] <= 1) and (ts[i] <= 1) and (rs[i] + ts[i] <= 1)):
                    results[i] = (1 - (rs[i] + ts[i]) , rs[i], ts[i])
        return results

    def IUV2FBC_fast( self, I_point , U_point, V_point):
        P = np.array([ U_point , V_point , 0 ])
        
        FaceIndicesNow  = np.where( self.FaceIndices == I_point )
        
        FacesNow = self.FacesDensePose[FaceIndicesNow]
        #
        P_0 = np.vstack( (self.U_norm[FacesNow][:,0], self.V_norm[FacesNow][:,0], np.zeros(self.U_norm[FacesNow][:,0].shape))).transpose()
        P_1 = np.vstack( (self.U_norm[FacesNow][:,1], self.V_norm[FacesNow][:,1], np.zeros(self.U_norm[FacesNow][:,1].shape))).transpose()
        P_2 = np.vstack( (self.U_norm[FacesNow][:,2], self.V_norm[FacesNow][:,2], np.zeros(self.U_norm[FacesNow][:,2].shape))).transpose()
        #
        bcs = self.barycentric_coordinates_fast(P_0, P_1, P_2, P)
        for i, bc in enumerate(bcs):
            if bc is not None:
                bc1,bc2,bc3 = bc
                return(FaceIndicesNow[0][i],bc1,bc2,bc3)
        #
        # If the found UV is not inside any faces, select the vertex that is closest!
        #
        D1 = scipy.spatial.distance.cdist( np.array( [U_point,V_point])[np.newaxis,:] , P_0[:,0:2]).squeeze()
        D2 = scipy.spatial.distance.cdist( np.array( [U_point,V_point])[np.newaxis,:] , P_1[:,0:2]).squeeze()
        D3 = scipy.spatial.distance.cdist( np.array( [U_point,V_point])[np.newaxis,:] , P_2[:,0:2]).squeeze()
        #
        minD1 = D1.min()
        minD2 = D2.min()
        minD3 = D3.min()
        #
        if((minD1< minD2) & (minD1< minD3)):
            return(  FaceIndicesNow[0][np.argmin(D1)] , 1.,0.,0. )
        elif((minD2< minD1) & (minD2< minD3)):
            return(  FaceIndicesNow[0][np.argmin(D2)] , 0.,1.,0. )
        else:
            return(  FaceIndicesNow[0][np.argmin(D3)] , 0.,0.,1. )

    def FBC2PointOnSurface( self, FaceIndex, bc1,bc2,bc3,Vertices ):
        ##
        Vert_indices = self.All_vertices[self.FacesDensePose[FaceIndex]]-1
        ##
        p = Vertices[Vert_indices[0],:] * bc1 +  \
            Vertices[Vert_indices[1],:] * bc2 +  \
            Vertices[Vert_indices[2],:] * bc3 
        ##
        return(p)
    
    def IUV2UV(self, I_point , U_point, V_point):
        #FBC = self.IUV2FBC(I_point , U_point, V_point)
        FBC = self.IUV2FBC_fast(I_point , U_point, 1-V_point)
        BC = FBC[1:]
        Vert_indices = self.All_vertices[self.FacesDensePose[FBC[0]]]-1
        uvs = get_uvs(Vert_indices)
        result = BC[0]*uvs[:,0]+BC[1]*uvs[:,1]+BC[2]*uvs[:,2]
        return np.array([result[0],1-result[1]])
    
    def IUV2UV2(self,i,u,v):
        u = int(u*256)
        v = int(v*256)
        return precomputed[(i-1)*256*256+u*256+v][1]
        
    def atlas_to_texture(self, atlas_im):
        texture = np.zeros((256, 256,3))
        size_i,size_u,size_v,_ = atlas_im.shape
        for i in range(size_i):
            print(i)
            for u in range(size_u):
                for v in range(size_v):
                    UV_coord = self.IUV2UV2(i+1,u/size_u,v/size_v)
                    texture[int(UV_coord[0]*255),int(UV_coord[1]*255),:]=\
                           atlas_im[i,v,u,:] 
        return texture.transpose([1,0,2])
    
def precompute_lookup():
    # Need to run to generate precomputed matrix used in IUV2UV2
    methods = DensePoseMethods()
    precomputed = []
    for i in range(1,25):
        for x in range(256):
            print(i,x)
            for y in range(256):
                precomputed.append([[i,x,y],
                        methods.IUV2UV(i,x/255,y/255)])
    pickle.dump(precomputed,open("precomputed.pkl","wb"))
    