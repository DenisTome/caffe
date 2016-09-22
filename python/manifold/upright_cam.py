# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:06:58 2016

@author: chrisr
"""
#import getdata as gd
import numpy as np
import segmentation as sg
from numpy.core.umath_tests import matrix_multiply
from upright_fast import reestimate_r, reestimate_r3d
# w, gt=gd.load_string('walking')
#import ceres
id3=np.identity(3)

# Used in the manifold layer
def estimate_r(xzdat, xzshape):
    """Reestimate r only from rest shape, without using basis coefficients"""
    u, _, _, _ = np.linalg.lstsq(xzshape.T, xzdat.T)
    norm = np.sqrt((u ** 2).sum(0))
    rot = u / norm
    return rot

# Used in the manifold layer
def estimate_a_and_r(w,e,s0,camera_r=False,Lambda=False):
    """So local optima are a problem in general.
    However:
        1. This problem is convex in a but not in r, and
        2. each frame can be solved independently.
    So for each frame, we can do a line search in r and take the globally 
    optimal solution.
    In practice, we just brute force over 100 different estimates of r, and take
    the best pair (r,a*(r)) where a*(r) is the optimal minimiser of a given r.
    """
    frames=w.shape[0]
    basis=e.shape[0]
    check=np.arange(0,1,0.01)*2*np.pi
    a=np.empty((check.size,frames,basis))
    res=np.empty((check.size,frames))
    for i in xrange(check.size):
        c=check[i]
        r=np.empty((2,w.shape[0]))
        r[0]=np.sin(c)
        r[1]=np.cos(c)
        a[i],res[i]=reestimate_a(w, e, s0, r,camera_r,Lambda)
    best=np.argmin(res,0)
    a=a[(best, np.arange(frames))]
    theta=check[best]
    r=np.empty((2,w.shape[0]))
    r[0]=np.sin(theta)
    r[1]=np.cos(theta)    
    return a,r

# Used in the manifold layer
def reestimate_a(w, e, s0, rot,camera_r=False,Lambda=False):
    """Reestimate a from rest shape, rotation, and basis coefficients
    as a least squares problem.
    solution minimises 
    ||W_i- P.dot(camera_r).dot(Mat(rot_i)).dot(s+a_i.e)||^2_2 + Lambda**2.dot(a**2)
    for each frame i. Where **2 is the elementwise square"""
    if camera_r is False:
        return reestimate_a_old(w,e,s0,rot)
    
    basis = e.shape[0]
    frames = w.shape[0]
    points = w.shape[-1]
    # co-ordinate frame is xzy
    # write P as short for the projection  P.dot(camera_r).dot(Mat(rot_i))
    mat_r = upgrade_r(rot.T).transpose(0,2,1)
    

    P= matrix_multiply(camera_r[np.newaxis,:2], mat_r)
    #For each frame project the shape and subtract that from the measurement matrix
    res=w-P.dot(s0)
    #vis.scatter2d(P.dot(s0)[0],w[0])
    res=res.reshape(frames,points*2)
    if Lambda is not False:
        res=np.hstack((res,np.zeros((frames,basis))))
    #compute the rotated e basis for each frame
    # output is frames,basis, 2, points
    # input is frames, 2,3 + basis,3,points
    re=np.einsum('ilk,jkp',P,e).reshape(frames,basis,2*points)
    re2=np.empty((frames,basis,2*points+basis))
    if Lambda is not False:
        re2[:,:,:2*points]=re
        re2[:,:,2*points:]=np.diag(Lambda)
        re=re2
    #Now solve for ||res-a.dot(re)||^2_2
    a=np.empty((frames,basis))
    a.fill(np.NaN)
    residue=np.empty(frames)
    for i in xrange(frames):
            #    if i ==0:
            #print target[i]
            #print re[i]
        a[i], b, _, _ = np.linalg.lstsq(re[i].T, res[i])
        residue[i]=b#.sum(1)
            #print aa
        #a[i]=aa
    #vis.scatter2d(P.dot(s0+(a[0,:,np.newaxis,np.newaxis]*e).sum(0))[0],w[0])
    return a,residue

# Used in the manifold layer
def reestimate_a_old(w, e, s0,rot):
    basis = e.shape[0]
    frames = w.shape[0]
    points = w.shape[-1]
    # for each frame find a that minimises the projection error
    
    #For each frame project the shape and subtract that from the measurement matrix
    yres = w[:,1] - s0[2]               # xzy format
    xzres = w[:,0] - rot.T.dot(s0[:2])  # take xz component and see how relevant they are
                                        # to find the u component (2d skeleton).
    target = np.concatenate((xzres,yres),-1)
    
    #For each frame project the entire basis
    rexz = np.einsum('ik,jkl', rot.T, e[:, :2]) #Frames,basis,points
    rey=np.empty((frames,basis,points))
    rey[:]=e[:, 2]
    re=np.concatenate((rexz,rey),-1)    # projectd bases; rexz -> u component; rey -> v component.


    # print frames
    # print (rexz.shape,rot.shape, e.shape,xzres.shape)
    # ok do the y componet as before
#    a, _, _, _ = np.linalg.lstsq(ey.T, yres.T)
    a=np.empty((frames,basis))
    a.fill(np.NaN)
    for i in xrange(frames):
            #    if i ==0:
            #print target[i]
            #print re[i]
        a[i], _, _, _ = np.linalg.lstsq(re[i].T, target[i])
            #print aa
        #a[i]=aa

    return a
#
#def error3d(gt,inv_rot,z,a,e):
#    rec = build_model(a,e,z)
#   # print inv_rot.shape,gt.shape
#    rec-=matrix_multiply(inv_rot,gt)
#    #rec[:,2]-=gt[:,2]
#    return (rec**2).sum()

# Used in the manifold layer
def upgrade_r(r):
    """Upgrades complex parameterisation of planar rotation to tensor containing
    per frame 3x3 rotation matrices"""
    assert(np.all(np.isfinite(r)))
    norm=np.sqrt((r[:,:2]**2).sum(1))
    assert(np.all(norm>0))
    r/=norm[:,np.newaxis]
    assert(np.all(np.isfinite(r)))
    newr = np.zeros((r.shape[0], 3, 3))
    newr[:, :2, 0] = r[:, :2]
    #newr[:, :2, 0] *= norm[:,np.newaxis]
    newr[:, 2, 2] = 1
    newr[:, 1::-1, 1] = r[:, :2]
    #newr[:, 1::-1, 1] *= norm[:,np.newaxis]
    newr[:,0, 1] *= -1
    return newr

# Used in the manifold layer
def build_model(a, e, s0=False):
    if s0 is not False:
        assert(s0.shape[0]==3)
    assert(e.shape[1]==3)
    assert(a.shape[1]==e.shape[0])
    out = a.dot(e.reshape(a.shape[1], -1)).reshape(a.shape[0], 3, -1)
    if s0 is False:
        return out
    else:
        return out + s0


# Used in the manifold layer
def project_model(mod, R_c, rot):
    """Projects model into the image plane using the planar camera rotation
    provided"""
    r2=upgrade_r(rot.T).transpose(0,2,1)
    proj=matrix_multiply(R_c[np.newaxis,:2],r2)
    return  matrix_multiply(proj,mod)
