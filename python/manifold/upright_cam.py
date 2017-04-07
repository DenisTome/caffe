# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:06:58 2016

@author: chrisr
"""
#import getdata as gd
import numpy as np
import segmentation as sg
from numpy.core.umath_tests import matrix_multiply
from upright_fast import (reestimate_r,  weighted_reestimate_r, reestimate_r3d,
                          estimate_a_and_r_with_depth,
                          pick_e,
                          estimate_a_and_r_with_weights,
                          estimate_a_and_r_with_scale,
                          estimate_a_and_r_with_weights_fixed_scale
                          )
# w, gt=gd.load_string('walking')
# import ceres
id3=np.identity(3)

    
# w= gd.load_ravi_w('Real_Seq/Face_Sequence')
# lab,w,gt=stack_strings(( 'drink', 'pickup', 'stretch', 'yoga'))
def estimate_rot_rigid(xzdat):
    """Initial estimate of rotation and shape using a planar orthographic camera,
    and affine relaxation based on standard Tomisi affine approach"""
    u, d, v = np.linalg.svd(xzdat, False)
    u = u[:, :2]
    v = v[:2]
    norm = np.sqrt((u ** 2).sum(1))
    #mnorm = norm.mean()
    #v *= d[:2, np.newaxis] * mnorm
    rot = u.T / norm
    v, _, _, _ = np.linalg.lstsq(rot.T, xzdat)

    return rot, v


def estimate_rot_rigid3d(xzdat):
    """Initial estimate of rotation and shape using a planar orthographic camera,
    and affine relaxation based on standard Tomisi affine approach"""
    assert(xzdat.shape[1] == 2)
    #xzdat = xzdat.copy()
    #xzdat[:, 1] = 0
    w = xzdat.reshape(-1, xzdat.shape[2])
    u, _, v = np.linalg.svd(w+np.random.rand(w.shape[0],w.shape[1])*10**-5, False)
    u = u[:, :2]
    v = v[:2]
    # p_rot, p_v = estimate_rot_rigid(w[::2])
    # u[:,]
    # frames = xzdat.shape[0]
    u[1::2] = 0
    u2 = np.empty((u.shape[0]/2, 2))
    u2[:, 0] = u[::2, 0] + u[1::2, 1]
    u2[:, 1] = u[::2, 1] - u[1::2, 0]
    norm = np.sqrt((u2 ** 2).sum(1))
    u2 /= norm[:, np.newaxis]
    u[::2] = u2
    u[1::2, ::-1] = u2
    u[1::2, 0] *= -1
    # Too much data. lstsq is failing to converge
    # Exploit R^-1 =R.T

    inv=np.empty_like(u)
    inv[::2] = u2
    inv[1::2, ::-1] = u2
    inv[::2, 1] *= -1
    inv = inv.reshape(-1, 2, 2) # .transpose(1, 0, 2)

    v = np.linalg.lstsq(u, xzdat.reshape(-1, xzdat.shape[2]))[0]
    #v = matrix_multiply(inv, xzdat).mean(0)
    return inv, u.T, v


def reestimate_rigid(xzdat, rot, weights=False):
    """Reestimate of shape using a planar orthographic camera, and known
    rotation"""
    if weights is False:
        v, _, _, _ = np.linalg.lstsq(rot.T, xzdat)
    else:
        v, _, _, _ = np.linalg.lstsq(rot.T*weights[:, np.newaxis],
                                     xzdat*weights[:, np.newaxis])
    return v


def estimate_r(xzdat, xzshape):
    """Reestimate r only from rest shape, without using basis coefficients"""
    u, _, _, _ = np.linalg.lstsq(xzshape.T, xzdat.T)
    norm = np.sqrt((u ** 2).sum(0))
    rot = u / norm
    return rot

def estimate_a_and_r(w,e,s0,camera_r=False,Lambda=False,weights=False):
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
    # call reestimate_a 100 times
    for i in xrange(check.size):
        c=check[i]
        r=np.empty((2,w.shape[0]))
        r[0]=np.sin(c)
        r[1]=np.cos(c)
        a[i],res[i]=reestimate_a(w, e, s0, r,camera_r,Lambda,weights)
    #find and return best coresponding solution
    best=np.argmin(res,0)
    a=a[(best, np.arange(frames))]
    theta=check[best]
    r=np.empty((2,w.shape[0]))
    r[0]=np.sin(theta)
    r[1]=np.cos(theta)    
    return a,r
    
def reestimate_a(w, e, s0, rot,camera_r=False,Lambda=False,weights=False):
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
    #print P[0]
    res=w-P.dot(s0)
    if weights is not False:
        res*=weights
    #vis.scatter2d(P.dot(s0)[0],w[0])
    res=res.reshape(frames,points*2)
    
    if Lambda is not False:
        res=np.hstack((res,np.zeros((frames,basis))))
    #compute the rotated e basis for each frame
    # output is frames,basis, 2, points
    # input is frames, 2,3 + basis,3,points  
    re=np.einsum('ilk,jkp',P,e).reshape(frames,basis,2*points)
    if weights is not False:
        re*=weights.reshape(weights.shape[0])[:,np.newaxis]
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


def planar_rec(w):
    """Generates 3d reconstruction based on affine relaxation\\
       subject to the constraint that the camera is planar, and alternating
       rows of w correspond to the y component of the reconstruction.\\
       Returns shape estimate and rotations parameterised as unit
       complex number."""
    y_init = w[1::2].mean(0)
    rot, sxz = estimate_rot_rigid(w[0::2])
    z_init = np.vstack((sxz[0], y_init, sxz[1]))

    return z_init, rot


def estimate_exz(a, rot, xzdat, v, weights=False, compactness=0):
    """estimate the xz component of e using rotation-matrix a coefficients and
    rest shape"""
    basis = a.shape[1]
    points = xzdat.shape[1]
    xzres = xzdat - rot.T.dot(v)
    # print (basis,frames,a.shape,rot.shape)
    ra = np.einsum('...j,...i', a, rot.T)
    # print ra.shape
    ra = ra.reshape(-1, basis * 2)
    # For each point find the optimal basis given coefficients a and
    # reprojection c
    if weights is not False:
        ra = ra * weights[:, np.newaxis]
        xzdat = xzres * weights[:, np.newaxis]

    if compactness:
        new_ra = np.zeros((ra.shape[0]*3, ra.shape[1]))
        new_ra[:ra.shape[0]] = ra
        if weights is False:
            new_ra[ra.shape[0]:2*ra.shape[0], :basis] = a * compactness
            new_ra[2*ra.shape[0]:, basis:] = a * compactness
        else:
            new_ra[ra.shape[0]: 2 * ra.shape[0], :basis] = (a * compactness *
                                                        weights[:, np.newaxis])

            new_ra[2*ra.shape[0]:, basis:] = (a * compactness *
                                              weights[:, np.newaxis])

        new_xzres = np.zeros((xzres.shape[0]*3, xzres.shape[1]))
        new_xzres[:xzres.shape[0]] = xzres
        exz, _, _, _ = np.linalg.lstsq(new_ra, new_xzres)
    else:
        exz, _, _, _ = np.linalg.lstsq(ra, xzres)

    # print exz.shape
    exz = exz.reshape(2, basis, points)
    return exz


def estimate_exz3d(a, rot, xzdat, v, weights=False, compactness=0):
    """estimate the xz component of e using rotation-matrix a coefficients and
    rest shape"""
    assert(xzdat.shape[1] == 2)
    basis = a.shape[1]
    points = xzdat.shape[2]
    xzres = xzdat.reshape(-1, xzdat.shape[-1]) - rot.T.dot(v)
    # print (basis,frames,a.shape,rot.shape)
    ra = np.einsum('...j,...hi', a, rot.T.reshape(-1, 2, 2))
    # print ra.shape
    ra = ra.reshape(-1, basis * 2)
    # For each point find the optimal basis given coefficients a and
    # reprojection c
    if weights is not False:
        ra = ra.reshape(-1, 2, basis * 2) * weights[:, np.newaxis,np.newaxis]
        xzdat = xzres.reshape(-1, 2, xzres.shape[-1]) * weights[:, np.newaxis, np.newaxis]
        ra = ra.reshape(-1, basis * 2)
        xzdat = xzdat.reshape(xzres.shape)
    if compactness:
        new_ra=np.zeros((ra.shape[0]*3,ra.shape[1]))
        new_ra[:ra.shape[0]]=ra
        if weights is False:
            new_ra[ra.shape[0]:2*ra.shape[0],:basis]=a*compactness
            new_ra[2*ra.shape[0]:,basis:]=a*compactness
        else:
            new_ra[ra.shape[0]:2*ra.shape[0],:basis]=a*compactness*weights[:,np.newaxis]

            new_ra[2*ra.shape[0]:,basis:]=a*compactness*weights[:,np.newaxis]

        new_xzres=np.zeros((xzres.shape[0]*3,xzres.shape[1]))
        new_xzres[:xzres.shape[0]]=xzres
        exz, _, _, _ = np.linalg.lstsq(new_ra, new_xzres)
    else:
        exz, _, _, _ = np.linalg.lstsq(ra, xzres)

    #print exz.shape
    exz = exz.reshape(2, basis, points)
    return exz

def estimate_a(yres, basis):
    """Takes residue in y component, i.e. y_dat-y_init and number of bases\\
    Returns a and y component of e"""
    a, d, ye = np.linalg.svd(yres, False)
    a = a[:, :basis]
    d = d[:basis]
    ye = ye[:basis]
    d = np.sqrt(d)
    a *= d
    ye = ye * d[:, np.newaxis]
    return a, ye

def estimate_a_from_e(w, e, s0,camera_r=id3):
    """Cold estimate of a from rest shape, and basis coefficients"""
    e2 = matrix_multiply(camera_r[np.newaxis,:,:],e)
    ey = e2[:, 1] # rotate bases according to the camera
    yres = w[:,1] - camera_r.dot(s0)[1] # removing y rest shape
    a, _, _, _ = np.linalg.lstsq(ey.T, yres.T)

    return a.T


def planar_rec_PCA(w, basis, r_guess=False):
    """Generates 3d reconstruction based on affine relaxation
       subject to the constraint that the camera is planar, and alternating
       rows of w correspond to the y component of the reconstruction.\\
       Add PCA_basis on top to capture deformations.\\
       As y is fully annotated use this to infer it's own basis and coefficients
       then transfer coefficients and rotations to infer x and z basis.
       """
    assert (basis > 0)

    ydat = w[1::2]
    xzdat = w[::2]
    y_init = ydat.mean(0)
    if r_guess is False:
        rot, v = estimate_rot_rigid(xzdat)
    else:
        rot = r_guess
        v = reestimate_rigid(xzdat, rot)

    z0 = np.vstack((v[0], y_init, v[1]))
    # Mean shape computed
    # find svd basis of y, truncate and split difference
    a, ye = estimate_a(ydat - y_init, basis)
    exz = estimate_exz(a, rot, xzdat, v)
    e = np.stack((exz[0], ye, exz[1])).transpose(1, 0, 2)
    return z0, rot, a, e


def planar_rec_PCA3d(z, basis, r_guess=False):
    """Generates 3d reconstruction based on affine relaxation
       subject to the constraint that the camera is planar, and alternating
       rows of w correspond to the y component of the reconstruction.\\
       Add PCA_basis on top to capture deformations.\\
       As y is fully annotated use this to infer it's own basis and
       coefficients
       then transfer coefficients and rotations to infer x and z basis.
       """
    assert (basis > 0)
    assert(z.shape[1] == 3)
    ydat = z[:, 2]
    xzdat = z[:, :2]
    y_init = ydat.mean(0)
    if r_guess is False:
        inv_r, rot, v = estimate_rot_rigid3d(xzdat)
    else:
        rot = r_guess
        v = reestimate_rigid(xzdat, rot) #probably still works

    z0 = np.vstack((v[0], y_init, v[1]))
    # Mean shape computed
    # find svd basis of y, truncate and split difference
    a, ye = estimate_a(ydat - y_init, basis) #still holds
    exz = estimate_exz3d(a, rot, xzdat, v)
    e = np.stack((exz[0], ye, exz[1])).transpose(1, 0, 2)
    return z0, rot, a, e

# ==============================================================================
# def planar_rec_error(w,s0,e,rho):
#     """Given preexisting pca basis, compute rotations and basis coefficients
#         as in planar_rec, and use this to induce unary costs
#        """
#     assert(s0.shape[0]==3)
#     assert(e.shape[1]==3)
#     points=s0.shape[1]
#     assert(w.shape[1]==points)
#     assert(e.shape[2]==points)
#     assert(w.shape[0]%2==0)
#
#     xzdat=w[::2]
#     xzshape=s0[::2]
#
#     rot= reestimate_r(xzdat, xzshape)
#
#
#     a,unary=reestimate_a(w_true,e_init,z_init,rot)
#
#     return unary
#
# ==============================================================================


def PCA(data, rank):
    mean = data.mean(0)
    res = data - mean
    u, d, v = np.linalg.svd(res, False)
    u = u[:, :rank]
    d = np.sqrt(d[:rank])
    v = v[:rank]

    return mean, u*d, v*d[:, np.newaxis]

def PCA2(data, rank):
    """PCA
    returns mu, a vector and unit e basis such that data ~ mu +a.e"""
    mean = data.mean(0)
    res = data - mean
    u, d, v = np.linalg.svd(res, False)
    u = u[:, :rank]
    d = d[:rank]
    v = v[:rank]

    return mean, u*d, v


def PPCA(data, rank):
    """Probalistic PCA
    returns mu, a vector and unit e basis such that data ~ mu +a.e
    and rank+1 weights describing 1./ sigma^2 estimators"""
    mean = np.average(data,0)
    res = data - mean
    cov=np.cov(res.T)
    assert(cov.shape[0]==data.shape[1])
    d,v=np.linalg.eig(cov)
    #u, d, v = np.linalg.svd(res*weights[:,np.newaxis], False)
    #u = u[:, :rank]
    d=np.real(d)
    v=np.real(v)
    v = v.T[:rank]
    sigma=d[:rank+1]
    sigma[-1]=d[rank:].mean()
    sigma=sigma**-0.5
    d = d[:rank]
    u=res.dot(v.T)
    return mean, u, v, sigma

def weighted_PPCA(data,weights, rank):
    """Weighted Probalistic PCA
    returns mu, a vector and unit e basis such that data ~ mu +a.e
    and rank+1 weights describing 1./ sigma^2 estimators"""
    mean = np.average(data,0,weights)
    res = data - mean
    cov=np.cov(res.T,aweights=weights)
    assert(cov.shape[0]==data.shape[1])
    d,v=np.linalg.eig(cov)
    v=np.real(v)
    d=np.real(d)
    #u, d, v = np.linalg.svd(res*weights[:,np.newaxis], False)
    #u = u[:, :rank]
    v = v.T[:rank]
    sigma=d[:rank+1]
    sigma[-1]=d[rank:].mean()
    sigma=sigma**-0.5
    d = d[:rank]
    u=res.dot(v.T)
    return mean, u, v, sigma


def GPCA(data, rank):
    """Gaussian PCA
    returns mu, a vector and unit e basis such that data ~ mu +a.e
    and rank+1 weights describing 1./ sigma^2 estimators
    Differs from PPCA in that it chooses final weight to preserve Gaussian 
    density else where """
    mean = data.mean(0)
    res = data - mean
    u, d, v = np.linalg.svd(res, False)
    u = u[:, :rank]
    weights=d[:rank+1]
    weights[-1]=np.sqrt((d[rank:]**2).mean())
    weights=weights**-1
    #d = d
    v = v[:rank]
    u*=d[:rank]
    return mean, u, v, d, weights

def get_basis(data, rank):
    mean = data.mean(0)
    res = data - mean
    u, d, v = np.linalg.svd(res, False)
    u = u[:, :rank]
    d = d[:rank]*np.sqrt(u**2).mean(0)
    v = v[:rank]

    return mean, v,d

def error3d(gt,inv_rot,z,a,e):
    rec = build_model(a,e,z)
   # print inv_rot.shape,gt.shape
    rec-=matrix_multiply(inv_rot,gt)
    #rec[:,2]-=gt[:,2]
    return (rec**2).sum()

def parameter3d_one_shot_old(dat,rank):
    """ Returns: inv_rot, z, a, e

    Conceptually working in 3d makes the whole thing much easier.
    We no long run the risk of overfitting, and do not have to solve the
    problem of
    reconstruction under incomplete data, as such the problem can be split into
    biconvex components.\n
    1. Estimate (inverse)rotations that map the data onto the model.\n
    2. Perform PCA on the data.\n
    \n
    However, there are certain properties that mean it's more advisable to
    perform carefully controlled initialisation, and not use
    parameter_refine3d. Instead we estimate rotations once for each rank
    by intialising with the previous frame. and we anneal the rank starting
    with 1 and slowly increasing it until it reaches the desired solution."""
    assert(rank >= 0)
    xz = estimate_rot_rigid3d(dat[:, :2])[2]
    z=np.empty((3, xz.shape[1]))
    z[::2] = xz
    z[1] = dat[:, 2].mean()
    frames = dat.shape[0]
    a = np.zeros((frames, 1))
    e = np.zeros((1, 3, dat.shape[2]))
    inv_rot = reestimate_r3d(dat[:, :2], z[::2], a, e[:, ::2]).transpose(1, 0, 2)
    shape = np.empty((frames, 3, dat.shape[-1]))

    for i in xrange(1, rank+1):
        for j in xrange(3):
            print ("1:" +str(error3d(dat,inv_rot,z,a,e)))
            matrix_multiply(inv_rot.transpose(2, 0, 1), dat[:, :2], shape[:, ::2])
            shape[:, 1] = dat[:, 2]
            mean, a, e = PCA(shape.reshape(frames, -1),i)
            z = mean.reshape(3, -1)
            e = e.reshape(-1, 3, z.shape[1])
            print ("2:"+str(error3d(dat,inv_rot,z,a,e)))
            if j ==-1: #Then reset rotations and start again from 0
                inv_rot = reestimate_r3d(dat[:, :2], z[::2], a, e[:, ::2]).transpose(1,0,2)
            else:
                inv_rot = reestimate_r3d(dat[:, :2], z[::2], a, e[:, ::2], inv_rot[:, 0]).transpose(1,0,2)
            #print a.shape, e.shape
    return z, a, e, inv_rot

def upgrade_r(r):
    """Upgrades complex parameterisation of planar rotation to tensor containing
    per frame 3x3 rotation matrices"""
    assert(r.ndim==2)
    #print r.shape
    assert(r.shape[1]==2) # Technically optional assert, but if this fails 
                          # Data is probably transposed
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

def normalise(e):
    """Orthonormalise baisis vectors so that e.T = e(dag)"""
    e=e.reshape(e.shape[0],-1)
    u,d,v=np.linalg.svd(e,0)
    return v

def new_rot(inv_rot,dat,a,e,z):
    """helper function for known_basis3d,and parameter3d_one_shot"""
 #       assert(np.all(dat[:,:2]!=0))
    r=inv_rot[:, :2,0]
    r=np.ascontiguousarray(r)
    assert(np.all(np.isfinite(r)))
    norm=(r[:,:2]**2).sum(1)
    assert(np.all(norm>0))
    assert(np.all(np.isfinite(dat)))
    m=build_model(a,e,z)
    assert(np.all(np.isfinite(m)))
#        assert(np.all(m[:,:2]!=0))
#        assert(np.all(m[:,2]==0))
    #vis.scatter3d(dat[0],m[0])
    ceres.planar_ICP(dat,np.ascontiguousarray(m),
                     r,solver_type=0,preconditioner=3,verbosity=0)

    return upgrade_r(r) #[:,(1,0,2)]

def known_basis2d(w, e, z, r=False, its=5):
    """Find inv_rot,a given e and z as fixed.
    Unclear if basis annealling is a good idea.
    Performs alternation based on estimating a then inv_rot (its) times"""
    assert(len(w.shape)==3)
    assert(z.shape[0] == 3)
    assert(e.shape[1] == 3)
    if r is not False:
        assert(r.shape[0] == 2)
        assert(r.shape[1] == w.shape[0])
        a=reestimate_a(w.reshape(-1,w.shape[-1]),e,z,r)
#    assert(a.shape[1] == e.shape[0])
    assert(w.shape[-1] == e.shape[-1])
    #a=np.zeros(w.shape[1],e.shape[0])

    if r is False:
        a=estimate_a_from_e(w,e,z)
        r=reestimate_r(w[0],z[::2],a,e[:,::2])
    for i in xrange(its):
        # cost_test(i*3)
        a=reestimate_a(w.reshape(-1,w.shape[-1]),e,z,r)
        r = reestimate_r(w[0], z[::2], a, e[:, ::2], r)
#        c=compactness
#        compactness=0
#        cost_test(i*3+3)
#        compactness=c
    if debug:
        return r, a, ret_cost
    return r, a


def known_basis3d(dat, e, z, inv_rot=False, its=5):
    """Find inv_rot,a given e and z as fixed.
    Unclear if basis annealling is a good idea.
    Performs alternation based on estimating a then inv_rot (its) times"""

    e2=e.reshape(e.shape[0],-1).T
    #e2=e2.T
    dat=np.ascontiguousarray(dat)
    shape = np.empty_like(dat)
    a = np.zeros((dat.shape[0], e.shape[0]))
    cost=[]
    if inv_rot is False:
        r=np.empty((dat.shape[0],3,3))
        r[:]=np.identity(3)
        cost.extend((error3d(dat,r,z,a,e),))
        inv_rot=new_rot (r,dat,a,e,z)

    for j in xrange(its):
        cost.extend((error3d(dat,inv_rot,z,a,e),))
        matrix_multiply(inv_rot, dat, shape)
        shape-=z
        assert(np.all(np.isfinite(shape)))
        shape_view=shape.reshape(dat.shape[0],-1)
        a=shape_view.dot(e2)
        cost.extend((error3d(dat,inv_rot,z,a,e),))
        inv_rot = new_rot(inv_rot,dat, a, e,z)
    return a, inv_rot, cost

def parameter3d_one_shot(dat,rank):
    """ Returns: inv_rot, z, a, e

    Conceptually working in 3d makes the whole thing much easier.
    We no long run the risk of overfitting, and do not have to solve the
    problem of
    reconstruction under incomplete data, as such the problem can be split into
    biconvex components.\n
    1. Estimate (inverse)rotations that map the data onto the model.\n
    2. Perform PCA on the data.\n
    \n
    However, there are certain properties that mean it's more advisable to
    perform carefully controlled initialisation, and not use
    parameter_refine3d. Instead we estimate rotations once for each rank
    by intialising with the previous frame. and we anneal the rank starting
    with 1 and slowly increasing it until it reaches the desired solution."""
    assert(rank >= 0)
#    import ceres,vis
    dat=np.ascontiguousarray(dat)

    #dat=dat[0:1]
    xz = estimate_rot_rigid3d(dat[:, :2])[2]
    z=np.empty((3, xz.shape[1]))
    z[:2] = xz
    z[2] = dat[:, 2].mean(0)
    #vis.scatter3d(z)
    frames = dat.shape[0]
    a = np.zeros((frames, 1))
    e = np.zeros((1, 3, dat.shape[2]))
    r=np.empty((dat.shape[0],3,3))
    r[:]=np.identity(3)
    print ("0:" +str(error3d(dat,r,z,a,e)))
    inv_rot=new_rot (r,dat,a,e,z)
    shape = np.empty((frames, 3, dat.shape[-1]))
    #d=dat.reshape(frames*3,dat.shape[-1])

    mask=np.ones_like(dat)
    for i in xrange(1, rank+1):
        for j in xrange(10):
            print (str(i)+" "+str(j)+" a: " +str(error3d(dat,inv_rot,z,a,e)))
            matrix_multiply(inv_rot, dat, shape)
            assert(np.all(np.isfinite(shape)))
            mean, a, e = PCA(shape.reshape(frames, -1),i)
            z = mean.reshape(3, -1)
            e = e.reshape(-1, 3, z.shape[1])
            print (str(i)+" "+str(j)+" b: "+str(error3d(dat,inv_rot,z,a,e)))
            inv_rot = new_rot(inv_rot,dat, a, e,z)
        zt=np.ascontiguousarray(z.T)
        ee=np.ascontiguousarray(e.transpose(2,0,1))
        r=np.ascontiguousarray(inv_rot.transpose(0,2,1)[:, :2,0])
        print (str(i)+" c*: " +str(error3d(dat,inv_rot,z,a,e)))
        # TODO: restore
        if i==1 : #or i==rank:
            ceres.ba_e3d(dat, mask, r, zt, a, ee,
                         solver_type=0,preconditioner=3,verbosity=0,model_type=2,
                         linear_solver=1)

        e=ee.transpose(1,2,0)
        inv_rot=upgrade_r(r).transpose(0,2,1)
        z=zt.T
        print (str(i)+" c: " +str(error3d(dat,inv_rot,z,a,e)))


    matrix_multiply(inv_rot, dat, shape)
    assert(np.all(np.isfinite(shape)))
    mean, a, e = PCA2(shape.reshape(frames, -1),rank)
    z = mean.reshape(3, -1)
    e = e.reshape(-1, 3, z.shape[1])
    print (str(i)+" d: " +str(error3d(dat,inv_rot,z,a,e)))


            #print a.shape, e.shape
    return z, a, e, inv_rot
    
def parameter3d_prob_one_shot(dat,rank):
    """ Returns: inv_rot, z, a, e

    Conceptually working in 3d makes the whole thing much easier.
    We no long run the risk of overfitting, and do not have to solve the
    problem of
    reconstruction under incomplete data, as such the problem can be split into
    biconvex components.\n
    1. Estimate (inverse)rotations that map the data onto the model.\n
    2. Perform PCA on the data.\n
    \n
    However, there are certain properties that mean it's more advisable to
    perform carefully controlled initialisation, and not use
    parameter_refine3d. Instead we estimate rotations once for each rank
    by intialising with the previous frame. and we anneal the rank starting
    with 1 and slowly increasing it until it reaches the desired solution."""
    assert(rank >= 0)
#    import ceres,vis
    dat=np.ascontiguousarray(dat)

    #dat=dat[0:1]
    xz = estimate_rot_rigid3d(dat[:, :2])[2]
    z=np.empty((3, xz.shape[1]))
    z[:2] = xz
    z[2] = dat[:, 2].mean(0)
    #vis.scatter3d(z)
    frames = dat.shape[0]
    a = np.zeros((frames, 1))
    e = np.zeros((1, 3, dat.shape[2]))
    r=np.empty((dat.shape[0],3,3))
    r[:]=np.identity(3)
    print ("0:" +str(error3d(dat,r,z,a,e)))
    inv_rot=new_rot (r,dat,a,e,z)
    shape = np.empty((frames, 3, dat.shape[-1]))
    #d=dat.reshape(frames*3,dat.shape[-1])

    mask=np.ones_like(dat)
    for i in xrange(1,1+rank):
        for j in xrange(10):
            #print str(i)+" "+str(j)+" a: " +str(error3d(dat,inv_rot,z,a,e))
            matrix_multiply(inv_rot, dat, shape)
            assert(np.all(np.isfinite(shape)))
            mean, a, e, sigma2 = PPCA(shape.reshape(frames, -1),i)
            z = mean.reshape(3, -1)
            e = e.reshape(-1, 3, z.shape[1])
            s=np.real(sigma2[:-1])**-2
            a*=(s/(0.005+s))
            print (str(i)+" "+str(j)+" b: "+str(error3d(dat,inv_rot,z,a,e)))
            inv_rot = new_rot(inv_rot,dat, a, e,z)
        #print str(i)+" c*: " +str(error3d(dat,inv_rot,z,a,e))
        # TODO: restore
        if False: #i==1 : #or i==rank:
            zt=np.ascontiguousarray(z.T)
            ee=np.ascontiguousarray(e.transpose(2,0,1))
            r=np.ascontiguousarray(inv_rot.transpose(0,2,1)[:, :2,0])

            ceres.ba_e3d(dat, mask, r, zt, a, ee,
                         solver_type=0,preconditioner=3,verbosity=0,model_type=2,
                         linear_solver=1)

            e=ee.transpose(1,2,0)
            inv_rot=upgrade_r(r).transpose(0,2,1)
            z=zt.T
            print (str(i)+" c: " +str(error3d(dat,inv_rot,z,a,e)))


    matrix_multiply(inv_rot, dat, shape)
    assert(np.all(np.isfinite(shape)))
    mean, a, e = PCA2(shape.reshape(frames, -1),rank)
    z = mean.reshape(3, -1)
    e = e.reshape(-1, 3, z.shape[1])
    print (str(i)+" d: " +str(error3d(dat,inv_rot,z,a,e)))


            #print a.shape, e.shape
    return z, a, e, inv_rot
    
def parameter_refine3d(dat, z, a, e, inv_rot, its=10,
                       weights=False, debug=False):
    """ Returns: inv_rot, z, a, e

    Conceptually working in 3d makes the whole thing much easier.
    We no long run the risk of overfitting, and do not have to solve the
    problem of
    reconstruction under incomplete data, as such the problem can be split into
    biconvex components.\n
    1. Estimate (inverse)rotations that map the data onto the model.
    2. Perform PCA on the data."""
    assert(weights is False)
    assert(z.shape[0] == 3)
    assert(e.shape[1] == 3)
    assert(a.shape[1] == e.shape[0])
    frames = dat.shape[0]
    rank=a.shape[1]
    assert(a.shape[0] == frames)
    assert(dat.shape[-1] == e.shape[-1])
    inv_rot = False #inv_rot.transpose(1,0,2)


    for i in xrange(its):
        print ("1:" +str(error3d(dat,inv_rot,z,a,e)))
        inv_rot = reestimate_r3d(dat[:, :2], z[::2], a, e[:, ::2], inv_rot[:, 0]).transpose(1,0,2)
        print ("2:"+str(error3d(dat,inv_rot,z,a,e)))
        shape = np.empty((frames, 3, dat.shape[-1]))
        matrix_multiply(inv_rot.transpose(2, 0, 1), dat[:, :2], shape[:, ::2])
        shape[:, 1] = dat[:, 2]
        #print a.shape, e.shape
        mean, a, e = PCA(shape.reshape(frames, -1),rank)
        z = mean.reshape(3, -1)
        e = e.reshape(-1, 3, z.shape[1])

    print ("3:"+str(error3d(dat,inv_rot,z,a,e)))

        #print a.shape, e.shape
    return inv_rot, z, a, e


def parameter_refine(xzdat, z, a, e, its=10, r=False, compactness=0,
                     weights=False, debug=False):
    assert(z.shape[0] == 3)
    assert(e.shape[1] == 3)
    if r is not False:
        assert(r.shape[0] == 2)
        assert(r.shape[1] == a.shape[0])
    assert(a.shape[1] == e.shape[0])
    assert(xzdat.shape[-1] == e.shape[-1])
    if r is False:
        r = reestimate_r(xzdat, z[::2], a, e[:, ::2])
    old_cost = np.asarray([np.Infinity])
    ret_cost = np.empty_like(weights)

    def cost_test(i):
        if debug:
            weight2 = weights ** 2
            inc = 0
            if compactness:
                m = build_model(a, e)
                mod = m + z
                m *= m
                if weights is not False:
                    inc = (m.reshape(m.shape[0], -1).sum(1).dot(weight2) *
                           compactness * compactness)
                else:
                    inc = m.sum() * compactness * compactness
            else:
                mod = build_model(a, e, z)
            p = project_model(mod, r)
            if weights is False:
                cost = ((xzdat - p[:, 0]) ** 2).sum()
            else:
                ret_cost[:] = ((xzdat - p[:, 0]) ** 2).sum(-1)
                ret_cost[:] += (m.reshape(m.shape[0], -1).sum(1) *
                                compactness * compactness)
                cost = ret_cost[:].dot(weight2)

#            print i, cost, inc,cost-inc
#            cost #+=inc
            if not (old_cost[0]*1.0001 >= cost):
                print (weights, compactness)
                assert False
     #       if i % 3:
            old_cost[0] = cost

    cost_test(-1)
    r = reestimate_r(xzdat, z[::2], a, e[:, ::2], r)
    for i in xrange(its):
        # cost_test(i*3)
        rae = np.einsum('...j,...i,ijk', r.T, a, e[:, 0::2])
        z[::2] = reestimate_rigid(xzdat - rae, r, weights)  # -rae
        cost_test(i*3+1)
        e[:, ::2] = estimate_exz(a, r, xzdat, z[::2], weights,
                                 compactness).transpose(1, 0, 2)
        r = reestimate_r(xzdat, z[::2], a, e[:, ::2], r)
        cost_test(i * 3 + 2)
#        c=compactness
#        compactness=0
#        cost_test(i*3+3)
#        compactness=c
    if debug:
        return r, z, e, ret_cost
    return r, z, e

def find_parameters(w_true, basis, global_r=False,compactness=0,its=5):
    """generates parameters describing a subset of the data w_true
       returns r,z,a,e"""
    z, r, a, e = planar_rec_PCA(w_true, basis, global_r)
    #    est1=build_model(a_init,e_init,z_init)
    #    a=reestimate_a(w_true,e_init,z_init,c_init)
    #    est2=build_model(a,e_init,z_init)
    xzdat = w_true[::2]
    parameter_refine(xzdat, z, a, e,r=r,compactness=compactness,its=its)
    return r, z, a, e


def build_model(a, e, s0=False):
    if s0 is not False:
        assert(s0.shape[0]==3)
    assert(e.shape[1]==3)
    assert(a.shape[1]==e.shape[0])
    out = a.dot(e.reshape(a.shape[1], -1)).reshape(a.shape[0], 3, -1)
    from getdata import renorm_gt as renorm

    if s0 is False:
        out,_=renorm(out)
        return out
    else:
        out,_=renorm(out+s0)

        return out



id3=np.identity(3)
def build_and_rot_model(a, e, s0, r):   
    r2=upgrade_r(r.T).transpose(0,2,1)
    mod = build_model(a, e, s0)
    mod=matrix_multiply(r2, mod)
    return mod


def project_model(mod, R_c, rot):
    """Projects model into the image plane using the planar camera rotation
    provided"""
    r2=upgrade_r(rot.T).transpose(0,2,1)
    proj=matrix_multiply(R_c[np.newaxis,:2],r2)
    return  matrix_multiply(proj,mod)


def make_rec(w_true, basis):
    z_init, c_init, a_init, e_init = planar_rec_PCA(w_true, basis)
    est = build_model(a_init, e_init, z_init)
    return est, c_init

def better_rec(w,est):
        est[:,:2]=w
        return est


def make_better_rec(w_true, basis):
    #z_init, c_init, a_init, e_init = planar_rec_PCA(w_true, basis)
    r, z, a, e = find_parameters(w_true, basis,compactness=0.1,its=5)
    est = build_and_rot_model(a, e, z,r)
    return better_rec(w_true, est), r




def make_rec2(w_true, basis):
    r, z, a, e = find_parameters(w_true, basis,compactness=0.1)
    est = build_model(a, e, z)
    return est, r


def build_nhood(coarse, size=10):
    """Build neighbourhood from initial coarse representation"""
    from sklearn.neighbors import NearestNeighbors
    coarse = coarse.reshape(coarse.shape[0], -1)
    nbrs = NearestNeighbors(n_neighbors=size + 1, algorithm='kd_tree',n_jobs=-1).fit(coarse)
    _, indices = nbrs.kneighbors(coarse)
    return indices  # [:,1:]


def estimate_parameters(w, s0, e,r=False):
    """Pipeline approach to compute good choice of a and r by exploiting problem
    structure. a is first computed using the y component. This is used to build
    a complete dynamic shape sequence and r is then estimated using truncated
    tomisi canadi"""
    ydat = w[1::2]
    xzdat = w[::2]
    # print (ydat.shape,e[:,1].shape)
    a, _, _, _ = np.linalg.lstsq(e[:, 1].T, (ydat - s0[1]).T)
    # print a.shape
    a = a.T
    r = reestimate_r(xzdat, s0[::2], a, e[:, ::2],r)
    return a, r


def make_unary(w, shapes, bases, compactness,
               global_r=False,
               per_r=False,
               per_a=False):
    points = w.shape[1]
    models = shapes.shape[0]
    frames = w.shape[0] / 2
    ww = w.reshape(2, frames, points).transpose(1, 0, 2)
    assert (bases.shape[0] == models)
    assert (shapes.shape[-1] == points)
    assert (bases.shape[-1] == points)
    unary = np.empty((models, frames))

    a = np.empty((models, frames,bases.shape[1]))
    r = np.empty((models, 2,frames))

    for i in xrange(models):
        aa, r[i] = estimate_parameters(w, shapes[i], bases[i])
        if per_a is not False:
            a[i]=per_a[i]
        else:
            a[i]=aa
        est = build_model(aa, bases[i], shapes[i])
        if global_r:
            r[i] = global_r
        if per_a is not False:
            r[i]=per_r[i]

        p = project_model(est, r[i])
        p -= ww
        p *= p
        unary[i] = p.sum(1).sum(1)
        if compactness:
            e = bases[i].reshape(-1, points * 3)
            ed = e.dot(e.T)
            dist = aa.dot(ed).dot(aa.T) * compactness
            unary[i] += np.diag(dist)
    return unary,a,r

def make_unary2(w,a,e,z,r,compactness=0):
    unary=np.empty((a.shape[0],a.shape[1]))
   # print w.shape,a.shape,e.shape,z.shape,r.shape
    assert(a.shape[0]==e.shape[0])
    assert(a.shape[0]==z.shape[0])
    if a.shape[0]!=r.shape[0]:
        print (a.shape, r.shape)
    assert(a.shape[0]==r.shape[0])

    #(714, 41) (105, 1) (179, 1, 3, 41) (179, 3, 41) (179, 2, 357)
    xzdat=w[::2]
    ydat=w[1::2]
    for i in xrange(a.shape[0]):
        mod=build_model(a[i],e[i],z[i])
    #    print mod.shape,r.shape
        p=project_model(mod,r[i])
     #   print p.shape
        #unary[i]= ((w.reshape(p.shape)-p)**2).sum(-1).sum(-1)
        unary[i]=((xzdat-p[:,0])**2).sum(-1)
        unary[i]=((ydat-p[:,1])**2).sum(-1)
        if compactness:
            mod=build_model(a[i],e[i])
            mod*=mod
            #mod[:,1]=0
            unary[i]+=mod.reshape(mod.shape[0],-1).sum(1)*compactness
    return unary

def weighted_reconstruction(unary, r, a, e, z, mag):
    models = unary.shape[0]
    # frames=unary.shape[1]
    unary = unary.min(1) - unary  # For stability reasons only
    unary *= mag
    ratio = np.exp(unary)
    ratio /= ratio.sum(1)
    est = np.zeros((models, 3, e.shape[-1]))
    for i in xrange(models):
        est[i] = build_model(a[i], e[i], z[i])

# gd.vis_manifold_rec(w,gt,est)
# gd.save_manifold(w,gt,est,filename="concatinate.mp4",subsample=1,fps=12)
# print gd.eval_func(gd.build_error_func(make_rec2))


def build_init(w, partitions, basis, global_r, compactness=0, its=5):
    points = w.shape[1]
    frames = w .shape[0]/2
    models = partitions.shape[0]
    # r0, z0, a0, e0 = find_parameters(w, basis)

    z = np.empty((models, 3, points))
    # z[:]=z0
    e = np.empty((models, basis, 3, points))
    # e[:]=e0
    a = np.empty((models, frames, basis))
    # a[:]=a0
    r = np.empty((models, 2, frames))
    r[:] = global_r
    ww = w.reshape(2, -1, points)
    # xzdat = ww[0]
    for i in xrange(models):
        # print i, partitions[i]
        r0, z[i], a0, e[i] = find_parameters(ww[:, partitions[i]].reshape(-1, points),
                                             basis, global_r[:, partitions[i]],
                                             compactness=0, its=its)
        r[i, :, partitions[i]] = r0.T
        a[i], r[i] = estimate_parameters(w, z[i], e[i], r=r[i])

    return z, e, a, r


# ==============================================================================
# mask,interior=sg.multi(unary.T.copy(),gnhood[:,:5].astype(np.double),
#                        np.ones(unary.shape[0])*0.0,
#                        unary.argmin(0).astype(np.double),
#                         exterior_weight=0.25)
#
# ==============================================================================
# vis.manifold_rec(w,gt,est,interior,False)

# vis.manifold_rec(w,gt,est,unary.argmin(0),False)
# vis.nhood_graph(est,gnhood[:,:5])
# vis.nhood_graph(est,gnhood[:,:5],unary.argmin(0),True)

def solve(w, s,a0, e, r, gnhood, MDL=0, ex_weight=0.25,
          visualise=False,iterations=5,DEBUG=False,compactness=0,checking=True):
    sq_comp=np.sqrt(compactness)
    unary=make_unary2(w,a0,e,s,r,compactness)
    basis = e.shape[1]
    xzdat = w[::2]
    ydat = w[1::2]

    def nonlocala(array):
        """This is the only case where the brokenness of python is actually a
        problem. hack round it using numpy references"""
        temp = np.empty(1, dtype=np.object)
        temp[0] = array
        return temp
    r=nonlocala(r)
    s = nonlocala(s)
    e = nonlocala(e)
    a=nonlocala(a0)
    #mask = nonlocal((False,))
#    interior = unary.argmin(0).astype(np.double)
    interior,upper,lower=sg.upper_bound(unary,gnhood,ex_weight)
    #mdlguess=MDL*np.unique(interior).size
    #print upper+mdlguess,lower+MDL,mdlguess

    interior=interior.astype(np.double)
    nhood = gnhood.astype(np.double)
    unary = nonlocala(unary)
    global old_m
    old_m=False


    def discrete(interior):
        multi,interior= sg.multi(unary[0].T.copy(), nhood,
                        np.ones(unary[0].shape[0]) * MDL,
                        interior,
                        exterior_weight=ex_weight)#0.25)  # 0.25)
        #if visualise:
            #import vis
            #vis.nhood_graphs(w, gt, est, gnhood[:, :5], interior, True)
        #print cost(multi)
        global old_m
        old_m=multi
        return multi,interior


    def cont(m, keep):
        xzdat=w[::2]
        s[0] = s[0][keep]
        e[0] = e[0][keep]
        r[0]=r[0][keep]
        a[0]=a[0][keep]
        mask = m >= ex_weight *.99
        #m[:]=mask
        unary[0] = unary[0][keep]
        #print -1,cost(m)

        #print cost(mask)
        # unary=unary[keep]
        # r=r[keep]
        for i in xrange(keep.shape[0]):
            if DEBUG:
                old_cost=cost(m)
            mm = mask[:, i]
            ss = s[0][i]
            #ydd = ydat[mm]
            weights=np.sqrt(m[mm,i])
            #ss[1] = np.average(ydd,0,weights)
            # a,ye=estimate_a(ydd-ss[1],basis)
            # e[0][i,:,1]=ye
            #w_trunc=w.reshape(2,ydat.shape[0],ydat.shape[1])[:,mm,:].reshape(-1,ydat.shape[1])
            #a =a0[keep[i]]# reestimate_a(w_trunc, e[0][i], ss, False)
            #print mm.sum()
            if mm.sum():
                #print str(i)+':', cost(new_mask)

                #e[0][i,:,::2]=estimate_exz(a,r[keep[i],:,mm].T,xzdat[mm],ss[::2],weights).transpose(1,0,2)
                if DEBUG:
                    r1,z1,e1,check=parameter_refine(xzdat[mm], ss.copy(),
                                          a[0][i,mm].copy(),
                                          e[0][i].copy(), its=10,
                                          r=r[0][i,:,mm].T.copy(),
                                          weights=weights,debug=DEBUG,
                                          compactness=sq_comp)
                else:
                    r1,z1,e1=parameter_refine(xzdat[mm], ss.copy(),
                                          a[0][i,mm].copy(),
                                          e[0][i].copy(), its=10,
                                          r=r[0][i,:,mm].T.copy(),
                                          weights=weights,debug=DEBUG,
                                          compactness=sq_comp)

                s[0][i]=z1
                e[0][i]=e1
                r[0][i,:,mm]=r1.T
                r[0][i] = reestimate_r(xzdat,ss[::2], a[0][i],
                                          e[0][i,:, ::2],r[0][i])
                r[0][i,:,mm]=r1.T
                #,compactness)               #,
                if DEBUG:
                    #unary[0],_,_ = make_unary(w, s[0], e[0], compactness,
                    #                          per_r=r[0],per_a=a[0])
                    new_mask=np.zeros_like(mask)
                    new_mask[mm,i]=m[mm,i]#m[:,i]
                    #unary[0]=make_unary2(w,a[0],e[0],s[0],r[0],0)
                    unary[0]=make_unary2(w,a[0],e[0],s[0],r[0],compactness)
                    cc=cost(new_mask)-MDL
                    print ("Values:",cc,cost(mask),cost(m),np.abs(check.sum()/cc))
                    if not (0.999<np.abs(check.dot(weights**2)/cc)<1.001):
                        print (weights,compactness)
                        assert False
                    print (cost(new_mask),cost(mask),cost(m))
                    print (i,old_cost,cost(m))
                    assert(old_cost+0.0001>=cost(m))

        unary[0]=make_unary2(w,a[0],e[0],s[0],r[0],compactness)
    #    unary[0],_,_ = make_unary(w, s[0], e[0], compactness)

    def cost(mask):
        return (unary[0] * mask.T).sum()+MDL*(np.any(mask,0)>0).sum()

    mask,interior=sg.standard_loop(discrete, cont, interior, cost,
                                   max_it=iterations,checking=checking)

    return mask, interior, s[0],a[0], e[0],r[0]



def hard_rec(interior,s,a,e):
    """performs reconstruction via hard assignment to interior models"""
    assert(interior.shape[0]==a.shape[0])
    interior=interior.astype(np.int)
    assert(interior.ndim==1)
    assert(s.ndim==3)
    assert(e.ndim==4)
    assert(a.ndim==2)
    s0=s[interior]
    e0=e[interior]
    out = np.einsum('...i,...ijk',a,e0)
    #print e.shape
    assert(out.shape==s0.shape)
    out+=s0
    from getdata import renorm_gt as renorm
    out,_=renorm(out)
    return out

def hard_rec_and_rotate(interior,s,a,e,r,camera_r=np.identity(3)):
    """performs reconstruction via hard assignment to interior models"""
    #print interior.shape, r.shape
    assert(interior.shape[0]==r.shape[1])
    s0=hard_rec(interior,s,a,e)
    r2=upgrade_r(r.T).transpose(0,2,1)
    #rot=matrix_multiply(camera_r[np.newaxis,:],r2)
    return matrix_multiply(r2, s0)
 