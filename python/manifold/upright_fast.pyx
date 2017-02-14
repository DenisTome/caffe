# -*- coding: utf-8 -*-
"""
upright_fast
cython replacement for bottleneck code in upright camera
Created on Thu Feb  4 15:09:43 2016

@author: chrisr
"""

import cython
cimport cython
import scipy.optimize as op
#cimport scipy.optimize as op
from cython.view cimport array as cvarray
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
import scipy.linalg
cimport scipy.linalg
from numpy.core.umath_tests import matrix_multiply
#from numpy.core.umath_tests cimport matrix_multiply

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)

def reestimate_r_old(np.ndarray[DTYPE_t, ndim=2]xzdat,
                     np.ndarray[DTYPE_t, ndim=2] xzshape,
                      np.ndarray[DTYPE_t, ndim=2] a,
                      np.ndarray[DTYPE_t, ndim=3] exz):
    """Reestimate r from rest shape, basis and basis coefficients"""
    dxz = a.dot(exz.reshape(a.shape[1], -1)).reshape(a.shape[0], 2, -1)
    dxz += xzshape
    # print (dxz.shape,xzdat.shape)
    frames = xzdat.shape[0]  #
    u = np.empty((2, frames))
    for i in xrange(frames):
        u[:, i], _, _, _ = np.linalg.lstsq(dxz[i].T, xzdat[i])
    norm = np.sqrt((u ** 2).sum(0))
    rot = u / norm
    return rot

def get_rotation(np.ndarray[DTYPE_t, ndim=2]w_total,
                 np.ndarray[DTYPE_t, ndim=3]shape_total,
                 r=False):
    if r is False:
        r=np.empty(w_total.shape[0]/2)
        r.fill(0) #Picked at random to give globally optimal results
    out=np.empty(w_total.shape[1])
    out2=np.empty(w_total.shape[1])
    w=np.empty(w_total.shape[1])
    shape=np.empty((shape_total.shape[1],shape_total.shape[2]))
    sin=np.empty(w_total.shape[1])
    cos=np.empty(w_total.shape[1])
    def f(theta):
        out[:]=w
        np.sin(theta,sin)
        np.cos(theta,cos)
        sin[:]*=shape[0]
        cos[:]*=shape[1]
        out[:]-=sin
        out[:]-=cos
        return out #.reshape(-1)
    def g(theta):
        np.sin(theta,out2)
        np.cos(theta,cos)
        cos[:]*=shape[0]
        out2[:]*=shape[1]
        out2[:]-=cos
        return out2
    def f_eval(theta):
        f(theta)
        out[:]*=out
        return out.sum()

    for i in xrange(w_total.shape[0]/2):
        w[:]=w_total[2*i]
        shape[:]=shape_total[i]
        r[i]=op.leastsq(func=f,x0=[r[i]],Dfun=g,full_output=False)[0]
        rt=op.leastsq(func=f,x0=[r[i]+np.pi],Dfun=g,full_output=False)[0]
        if f_eval(r[i])>f_eval(rt):
            r[i]=rt
    return r

# not used anymore, just for checking the code
def get_rotation_funky_offset_old(np.ndarray[DTYPE_t, ndim=2]w_total,
                 np.ndarray[DTYPE_t, ndim=3]shape_total,
                 r_init=False):

    if r_init is False:
        r=np.empty(w_total.shape[0])
        r.fill(0) #Picked at random to give globally optimal results
    else:
        r=r_init

    # since we consider just planar motion, we fix y and we check the rotation 
    # that best aligns the u component. This is why w is (17,)
    out=np.empty(w_total.shape[1])
    out2=np.empty(w_total.shape[1])
    w=np.empty(w_total.shape[1])
    shape=np.empty((shape_total.shape[1],shape_total.shape[2]))
    sin=np.empty(w_total.shape[1])
    cos=np.empty(w_total.shape[1])
    def f(theta):
        out[:]=w
        sin[:]=np.sin(theta)
        cos[:]=np.cos(theta)
        sin[:]*=shape[0]
        cos[:]*=shape[1]
        out[:]-=sin
        out[:]-=cos
        return out #.reshape(-1)
    def g(theta):
        out2[:]=np.sin(theta)
        cos[:]=np.cos(theta)
        cos[:]*=shape[0]
        out2[:]*=shape[1]
        out2[:]-=cos
        return out2
    def f_eval(theta):
        f(theta)
        out[:]*=out
        return out.sum()
        
    tot=0
    old_tot=0
    check=np.arange(8)*np.pi/4.0
    for i in xrange(w_total.shape[0]):
        w=w_total[i]
        shape[:]=shape_total[i]
        prev=f_eval(r[i])

        for c in check:
            if f_eval(r[i])>f_eval(c):
                r[i]=c
        if (i!=0) and (f_eval(r[i])>f_eval(r[i-1])):
            r[i]=r[i-1]
        r[i]=op.leastsq(func=f,x0=r[i:i+1],Dfun=g,full_output=False)[0]
        curr=f_eval(r[i])
        tot+=curr
        old_tot+=prev
        if prev<curr:
            print i, prev,curr
            assert(prev+0.0001>=curr)

    print "inner est cost:"+str(old_tot)+" --> " +str(tot)

    return r


def get_rotation_funky_offset(np.ndarray[DTYPE_t, ndim=3]w_total,
                 np.ndarray[DTYPE_t, ndim=3]shape_total,
                 np.ndarray[DTYPE_t, ndim=2]R_cam, 
                 r_init=False):

    if r_init is False:
        r=np.empty(w_total.shape[0])
        r.fill(0) #Picked at random to give globally optimal results
    else:
        r=r_init

    out=np.empty([2, w_total.shape[2]])
    out2=np.empty([2, w_total.shape[2]])
    o=out.reshape(-1)
    o2=out2.reshape(-1)
    w=np.empty([2, w_total.shape[2]])
    shape=np.empty((shape_total.shape[1],shape_total.shape[2]))
    #sin=np.empty(w_total.shape[1])
    #cos=np.empty(w_total.shape[1])

    # be careful, since xzy format
    
    PR=-R_cam[:2]
    R_i = np.zeros([3,3])
    R_i_p = np.zeros([3,3])
    R_i[2,2]   = 1
    
    R_cache=np.empty((2,3))
    
    def f(theta):
        # rotation compliant to the xzy format    
        #R_i[0, :2]  = [np.sin(theta) , np.cos(theta)]
        #R_i[1, :2] = [-np.cos(theta), np.sin(theta)]
        R_i[0, 0]  = np.sin(theta) 
        R_i[0, 1]  = np.cos(theta)
        R_i[1, 0]  = -R_i[0,1]
        R_i[1, 1]  = R_i[0,0]
        PR.dot(R_i,R_cache)
        R_cache.dot(shape,out)
        out[:]+=w
        return o
    def g(theta):
        #R_i_p[0, :2]  = [np.cos(theta), -np.sin(theta)]
        #R_i_p[1, :2] = [np.sin(theta), np.cos(theta)]
        R_i_p[0, 0]  = np.cos(theta) 
        R_i_p[1, 0]  = np.sin(theta)
        R_i_p[0, 1]  = -R_i[1,0]
        R_i_p[1, 1]  = R_i[0,0]        
        PR.dot(R_i_p,R_cache)
        R_cache.dot(shape,out2)
        return o2
    def f_eval(theta):
        f(theta)
        out[:]*=out
        return out.sum()

    tot=0
    old_tot=0
    check=np.arange(8)*np.pi/4
    for i in xrange(w_total.shape[0]):
        w=w_total[i]
        shape[:]=shape_total[i]
        prev=f_eval(r[i])

        for c in check:
            if f_eval(r[i])>f_eval(c):
                r[i]=c
        if (i!=0) and (f_eval(r[i])>f_eval(r[i-1])):
            r[i]=r[i-1]
        r[i]=op.leastsq(func=f,x0=r[i:i+1],Dfun=g,full_output=False)[0]
        curr=f_eval(r[i])
        tot+=curr
        old_tot+=prev
        #if prev<curr:
        #    print i, prev,curr
        assert(prev+0.0001>=curr)

    #print "inner est cost:"+str(old_tot)+" --> " +str(tot)

    return r
    
def weighted_get_rotation_funky_offset(np.ndarray[DTYPE_t, ndim=3]w_total,
                                       np.ndarray[DTYPE_t, ndim=3]shape_total,
                                       np.ndarray[DTYPE_t, ndim=2]R_cam, 
                                       np.ndarray[DTYPE_t, ndim=3]weights,
                                       r_init=False):

    if r_init is False:
        r=np.empty(w_total.shape[0])
        r.fill(0) #Picked at random to give globally optimal results
    else:
        r=r_init

    out=np.empty([2, w_total.shape[2]])
    out2=np.empty([2, w_total.shape[2]])
    o=out.reshape(-1)
    o2=out2.reshape(-1)
    w=np.empty([2, w_total.shape[2]])
    shape=np.empty((shape_total.shape[1],shape_total.shape[2]))
    #sin=np.empty(w_total.shape[1])
    #cos=np.empty(w_total.shape[1])
    weight=np.empty([2, w_total.shape[2]])
    # be careful, since xzy format
    
    PR=-R_cam[:2]
    R_i = np.zeros([3,3])
    R_i_p = np.zeros([3,3])
    R_i[2,2]   = 1
    
    R_cache=np.empty((2,3))
    
    def f(theta):
        # rotation compliant to the xzy format    
        #R_i[0, :2]  = [np.sin(theta) , np.cos(theta)]
        #R_i[1, :2] = [-np.cos(theta), np.sin(theta)]
        R_i[0, 0]  = np.sin(theta) 
        R_i[0, 1]  = np.cos(theta)
        R_i[1, 0]  = -R_i[0,1]
        R_i[1, 1]  = R_i[0,0]
        PR.dot(R_i,R_cache)
        R_cache.dot(shape,out)
        out[:]+=w
        out[:]*=weight
        return o
    def g(theta):
        #R_i_p[0, :2]  = [np.cos(theta), -np.sin(theta)]
        #R_i_p[1, :2] = [np.sin(theta), np.cos(theta)]
        R_i_p[0, 0]  = np.cos(theta) 
        R_i_p[1, 0]  = np.sin(theta)
        R_i_p[0, 1]  = -R_i[1,0]
        R_i_p[1, 1]  = R_i[0,0]        
        PR.dot(R_i_p,R_cache)
        R_cache.dot(shape,out2)
        out2[:]*=weight
        return o2
    def f_eval(theta):
        f(theta)
        out[:]*=out
        return out.sum()

    tot=0
    old_tot=0
    check=np.arange(8)*np.pi/4
    for i in xrange(w_total.shape[0]):
        w=w_total[i]
        shape[:]=shape_total[i]
        prev=f_eval(r[i])
        weight=weights[i]
        for c in check:
            if f_eval(r[i])>f_eval(c):
                r[i]=c
        if (i!=0) and (f_eval(r[i])>f_eval(r[i-1])):
            r[i]=r[i-1]
        r[i]=op.leastsq(func=f,x0=r[i:i+1],Dfun=g,full_output=False)[0]
        curr=f_eval(r[i])
        tot+=curr
        old_tot+=prev
        #if prev<curr:
        #    print i, prev,curr
        assert(prev+0.0001>=curr)

    #print "inner est cost:"+str(old_tot)+" --> " +str(tot)

    return r

def get_rotation_funky_offset3d(np.ndarray[DTYPE_t, ndim=3]w_total,
                 np.ndarray[DTYPE_t, ndim=3]shape_total,
                 r_init=False):

    if r_init is False:
        r=np.empty(w_total.shape[0])
        r.fill(0) # Makes camera aligned to first frame
    else:
        r=r_init
    #print (w_total.shape[0],w_total.shape[1],w_total.shape[2])
    #print (shape_total.shape[0],shape_total.shape[1],shape_total.shape[2])
    assert(w_total.shape[0]==shape_total.shape[0])
    assert(w_total.shape[1]==shape_total.shape[1])
    assert(w_total.shape[2]==shape_total.shape[2])
    out=np.empty((2,w_total.shape[2]))
    out2=np.empty((2,w_total.shape[2]))
    w=0
    shape=np.empty((2,shape_total.shape[2]))
    sin=np.empty((2,w_total.shape[2]))
    cos=np.empty((2,w_total.shape[2]))
    out_view=out.reshape(-1)
    out2_view=out2.reshape(-1)
    #np.sin(r,rot[0,0])
    #np.cos(r,rot[0,1])
    #rot[1,0]=-rot[0,1]
    #rot[1,1]=rot[0,0]

    def f(theta):
        out[:]=w
        sin[0,:]=shape[0]
        sin[1,:]=shape[1]
        # sin2[:]*=-1
        cos[0,:]=shape[1]
        cos[1,:]=shape[0]

        sin[:]*=np.sin(theta)
        cos[:]*=np.cos(theta)

        out[0]-=sin[0]
        out[0]-=cos[0]

        out[1]-=sin[1]
        out[1]+=cos[1]
        return out_view #.reshape(-1)

    def g(theta):
        out2[0]=shape[1]
        out2[1]=shape[0]
        out2[:]*=np.sin(theta)



        cos[0,:]=shape[0]
        cos[1,:]=shape[1]
        cos[:]*=np.cos(theta)

        out2[0]+=cos[0]
        out2[1]-=cos[1]
        return out2_view

    def f_eval(theta):
        f(theta)
        out[:]*=out
        return out.sum()

    tot=0
    old_tot=0
    #print w_total.shape[0],w_total.shape[1],w_total.shape[2]
    check=np.arange(8)*np.pi/4.0
    for i in xrange(w_total.shape[0]):
        w=w_total[i]
        shape[:]=shape_total[i]
        prev=f_eval(r[i])

        for c in check:
            if f_eval(r[i])>f_eval(c):
                r[i]=c
        if (i!=0) and (f_eval(r[i])>f_eval(r[i-1])):
            r[i]=r[i-1]
        r[i]=op.leastsq(func=f,x0=r[i:i+1],Dfun=g,full_output=False)[0]
        curr=f_eval(r[i])
        tot+=curr
        old_tot+=prev
        #if prev<curr:
        #    print i, prev,curr
        #    assert(prev>=curr)

    print "inner est cost:"+str(old_tot)+" --> " +str(tot)

    return r

def reestimate_r3d(np.ndarray[DTYPE_t, ndim=3]xzdat,
                 np.ndarray[DTYPE_t, ndim=2] xzshape,
                 np.ndarray[DTYPE_t, ndim=2] a,
                 np.ndarray[DTYPE_t, ndim=3] exz,r_init=False):
    """Reestimate r from rest shape, basis and basis coefficients"""
    if r_init is not False:
        assert(r_init.shape[0]==2)
        assert(r_init.shape[1]==a.shape[0])

    dxz = a.dot(exz.reshape(a.shape[1], -1)).reshape(a.shape[0], 2, -1)
    dxz += xzshape
    if r_init is not False:
        r=np.arccos(r_init[1])
        r*=np.sign(r_init[0])
    else:
        r=False

    # print (dxz.shape,xzdat.shape)
    from numpy.core.umath_tests import matrix_multiply
    #if r_init is not False:
   #     print (r_init.shape[0],r_init.shape[1],r_init.shape[2])
   #     print (xzdat.shape[0],xzdat.shape[1],xzdat.shape[2])
    #    print "init cost:"+str(((dxz-matrix_multiply(r_init,xzdat))**2).sum())
    r=get_rotation_funky_offset3d(xzdat,dxz,r)
    rot=np.empty((2,2,xzdat.shape[0]))
    #print r.shape[0]
    np.sin(r,rot[0,0])
    np.cos(r,rot[0,1])
    rot[1,0]=-rot[0,1]
    rot[1,1]=rot[0,0]

    #temp=xzdat.copy()
    #temp-=matrix_multiply(rot.transpose(2,0,1),dxz)
    #print "final cost:"+str((temp**2).sum())
    return rot

# not used anymore, just for checking the code
def reestimate_r_old(np.ndarray[DTYPE_t, ndim=2]xzdat,
                 np.ndarray[DTYPE_t, ndim=2] xzshape,
                 np.ndarray[DTYPE_t, ndim=2] a,
                 np.ndarray[DTYPE_t, ndim=3] exz,r_init=False):
    """Reestimate r from rest shape, basis and basis coefficients"""
    # xzdat is just the u component
    # data that uniquely describes the position of x and z coordinates
    if r_init is not False:
        assert(r_init.shape[0]==2)
        assert(r_init.shape[1]==a.shape[0])

    dxz = a.dot(exz.reshape(a.shape[1], -1)).reshape(a.shape[0], 2, -1)
    dxz += xzshape
    if r_init is not False:
        r_init=np.arccos(r_init[1])*np.sign(r_init[0])

    # print (dxz.shape,xzdat.shape)
    # dxz is the skeleton considering only the x and z component
    # since this is a planar rotation we are not considering y
    r=get_rotation_funky_offset_old(xzdat,dxz,r_init)
    rot=np.empty((2,xzdat.shape[0]))
    np.sin(r,rot[0])
    np.cos(r,rot[1])
    return rot

def reestimate_r(np.ndarray[DTYPE_t, ndim=3] xzydat,
                 np.ndarray[DTYPE_t, ndim=2] xzyshape,
                 np.ndarray[DTYPE_t, ndim=2] a,
                 np.ndarray[DTYPE_t, ndim=3] exzy,
                 np.ndarray[DTYPE_t, ndim=2] R_c, r_init=False):
    """Reestimate r from rest shape, basis and basis coefficients"""
    # xzdat is just the u component
    # data that uniquely describes the position of x and z coordinates
    if r_init is not False:
        assert(r_init.shape[0]==2)
        assert(r_init.shape[1]==a.shape[0])
    
    dxzy = a.dot(exzy.reshape(a.shape[1], -1)).reshape(a.shape[0], 3, -1)
    dxzy += xzyshape
    if r_init is not False:
        r_init=np.arccos(r_init[1])*np.sign(r_init[0])

    # print (dxz.shape,xzdat.shape)
    # dxz is the skeleton considering only the x and z component
    # since this is a planar rotation we are not considering y
    r=get_rotation_funky_offset(xzydat,dxzy,R_c,r_init)
    rot=np.empty((2,xzydat.shape[0]))
    np.sin(r,rot[0])
    np.cos(r,rot[1])
    return rot

def weighted_reestimate_r(np.ndarray[DTYPE_t, ndim=3] xzydat,
                          np.ndarray[DTYPE_t, ndim=3] weights,
                          np.ndarray[DTYPE_t, ndim=2] xzyshape,
                          np.ndarray[DTYPE_t, ndim=2] a,
                          np.ndarray[DTYPE_t, ndim=3] exzy,
                          np.ndarray[DTYPE_t, ndim=2] R_c, r_init=False):
    """Reestimate r from rest shape, basis and basis coefficients"""
    # xzdat is just the u component
    # data that uniquely describes the position of x and z coordinates
    if r_init is not False:
        assert(r_init.shape[0]==2)
        assert(r_init.shape[1]==a.shape[0])
    
    dxzy = a.dot(exzy.reshape(a.shape[1], -1)).reshape(a.shape[0], 3, -1)
    dxzy += xzyshape
    if r_init is not False:
        r_init=np.arccos(r_init[1])*np.sign(r_init[0])

    # print (dxz.shape,xzdat.shape)
    # dxz is the skeleton considering only the x and z component
    # since this is a planar rotation we are not considering y
    r=weighted_get_rotation_funky_offset(xzydat,weights,dxzy,R_c,r_init)
    rot=np.empty((2,xzydat.shape[0]))
    np.sin(r,rot[0])
    np.cos(r,rot[1])
    return rot

def pick_e(np.ndarray[DTYPE_t, ndim=3] w,
           np.ndarray[DTYPE_t, ndim=4] e,
           np.ndarray[DTYPE_t, ndim=3] s0,
           np.ndarray[DTYPE_t, ndim=2] camera_r=np.asarray([[1,0,0],
                                                            [0,0,-1],
                                                            [0,1,0]]),
           np.ndarray[DTYPE_t, ndim=2] Lambda=np.ones((0,0)),
           np.ndarray[DTYPE_t, ndim=3] weights=np.ones((0,0,0)),
           DTYPE_t scale_prior=-0.0014,                     
           DTYPE_t interval=0.01,DTYPE_t depth_reg=0.0325):
    """Brute force over charts from the manifold to find the best one.
        Returns best chart index and its a and r coefficients
        Returns assignment, and a and r coefficents"""
    charts=e.shape[0]
    frames=w.shape[0]
    basis=e.shape[1]
    points=e.shape[3]
    assert(s0.shape[0]==charts)    
    r=np.empty((charts,2,frames))
    a=np.empty((charts,frames,e.shape[1]))
    score=np.empty((charts,frames))
    check=np.arange(0,1,interval)*2*np.pi
    cache_a=np.empty((check.size,basis,frames))
    residue=np.empty((check.size,frames))
    
    if (Lambda.size!=0):
        res=np.zeros((frames,points*2+basis+points))
        proj_e=np.zeros((basis,2*points+basis+points))
        #d/=Lambda[Lambda.shape[0]-1]
    else:
        res=np.empty((frames,points*2))
        proj_e=np.empty((basis,2*points))
    Ps=np.empty((2,points))
    
    if weights.size==0:
        for i in xrange(charts):
            if Lambda.size!=0:
                a[i], r[i], score[i]=estimate_a_and_r_with_res(w,e[i],
                                                        s0[i],camera_r,Lambda[i],
                                                        check,cache_a,weights,
                                                        res,proj_e,residue,Ps,
                                                        depth_reg,scale_prior)
            else:
                a[i], r[i], score[i]=estimate_a_and_r_with_res(w,e[i],
                                                        s0[i],camera_r,Lambda,
                                                        check,cache_a,weights,
                                                        res,proj_e,residue,Ps,
                                                        depth_reg,scale_prior)
    else:
        w2=weights.reshape(weights.shape[0],-1)
        for i in xrange(charts):
            if Lambda.size!=0:
                a[i], r[i], score[i]=estimate_a_and_r_with_res_weights(w,e[i],
                                                        s0[i],camera_r,Lambda[i],
                                                        check,cache_a,w2,
                                                        res,proj_e,residue,Ps,
                                                        depth_reg,scale_prior)
            else:
                a[i], r[i], score[i]=estimate_a_and_r_with_res_weights(w,e[i],
                                                        s0[i],camera_r,Lambda,
                                                        check,cache_a,w2,
                                                        res,proj_e,residue,Ps,
                                                        depth_reg,scale_prior)
    
    remaining_dims=3*w.shape[2]-e.shape[1]
    assert(np.all(score>0))
    assert(remaining_dims>=0)
    llambda=-np.log(Lambda)
    lgdet=np.sum(llambda[:,:-1],1)+llambda[:,-1]*remaining_dims
    score/=2
    #score+=lgdet[:,np.newaxis] #Check me

    #best=np.argmin(score,0)
    #index=(best, np.arange(frames))
    #a=a[index]
    #r=r.transpose(0,2,1)[index].T
    
    return  score,a,r#[0]
        

def estimate_a_and_r(np.ndarray[DTYPE_t, ndim=3] w,
                     np.ndarray[DTYPE_t, ndim=3] e,
                     np.ndarray[DTYPE_t, ndim=2] s0,
                     np.ndarray[DTYPE_t, ndim=2] camera_r=np.asarray([[1,0,0],
                                                                      [0,0,-1],
                                                                      [0,1,0]]),
                     np.ndarray[DTYPE_t, ndim=1] Lambda=np.ones(0),
                     np.ndarray[DTYPE_t, ndim=3] weights=np.ones((0,0,0)),
                     DTYPE_t interval=0.01):
    """So local optima are a problem in general.
    However:
    
        1. This problem is convex in a but not in r, and
        
        2. each frame can be solved independently.
        
    So for each frame, we can do a grid search in r and take the globally 
    optimal solution.
    
    In practice, we just brute force over 100 different estimates of r, and take
    the best pair (r,a*(r)) where a*(r) is the optimal minimiser of a given r.
    
    Arguments:
    
        w is a 3d measurement matrix of form frames*2*points
        
        e is a 3d set of basis vectors of from basis*3*points
        
        s0 is the 3d rest shape of form 3*points
        
        Lambda are the regularisor coefficients on the coefficients of the weights
        typically generated using PPCA
        
        interval is how far round the circle we should check for break points
        we check every interval*2*pi radians
        
    Returns:
    
        a (basis coefficients) and r (representation of rotations as a complex 
        number)
    """
    frames=w.shape[0]
    points=w.shape[2]
    basis=e.shape[0]
    check=np.arange(0,1,interval)*2*np.pi
    a=np.empty((check.size,basis,frames))
    residue=np.empty((check.size,frames))
    # call reestimate_a 100 times
    r=np.empty(2)
    if (Lambda.size!=0):
        res=np.zeros((frames,points*2+basis))
        proj_e=np.zeros((basis,2*points+basis))
        d=np.diag(Lambda)
    else:
        res=np.empty((frames,points*2))
        proj_e=np.empty((basis,2*points))
        
    for i in xrange(check.size):
        c=check[i]
        r[0]=np.sin(c)
        r[1]=np.cos(c)
        rot=camera_r[:2].dot(upgrade_r(r).T)
        Ps=rot.dot(s0)
        res[:,:points*2]=w.reshape((frames,points*2))
        res[:,:points*2]-=Ps.reshape(points*2)
        if (Lambda.size!=0):
            proj_e[:,2*points:]=d
            res[:,2*points:].fill(0)
        if weights.size!=0: 
            res[:,:points*2]*=weights
        proj_e[:,:2*points]=rot.dot(e).transpose(1,0,2).reshape(e.shape[0],2*points)
        a[i], residue[i], _, _ = scipy.linalg.lstsq(proj_e.T, res.T,
                                                             overwrite_a=True,
                                                             overwrite_b=True)
    #find and return best coresponding solution
    best=np.argmin(residue,0)
    assert(best.shape[0]==frames)
    theta=check[best]
    a=a.transpose(0,2,1)[(best,np.arange(frames))]
    #residue=residue[index]
    r=np.empty((2,frames))
    r[0]=np.sin(theta)
    r[1]=np.cos(theta)    
    return a,r

    
def estimate_a_and_r_with_depth(np.ndarray[DTYPE_t, ndim=3] w,
                     np.ndarray[DTYPE_t, ndim=3] e,
                     np.ndarray[DTYPE_t, ndim=2] s0,
                     np.ndarray[DTYPE_t, ndim=2] camera_r=np.asarray([[1,0,0],
                                                                      [0,0,-1],
                                                                      [0,1,0]]),
                     np.ndarray[DTYPE_t, ndim=1] Lambda=np.ones(0),
                     DTYPE_t interval=0.01,                     
                     DTYPE_t depth_reg=0):
    """ Now with depth regularisation.
    So local optima are a problem in general.
    However:
    
        1. This problem is convex in a but not in r, and
        
        2. each frame can be solved independently.
        
    So for each frame, we can do a grid search in r and take the globally 
    optimal solution.
    
    In practice, we just brute force over 100 different estimates of r, and take
    the best pair (r,a*(r)) where a*(r) is the optimal minimiser of a given r.
    
    Arguments:
    
        w is a 3d measurement matrix of form frames*2*points
        
        e is a 3d set of basis vectors of from basis*3*points
        
        s0 is the 3d rest shape of form 3*points
        
        Lambda are the regularisor coefficients on the coefficients of the weights
        typically generated using PPCA
        
        interval is how far round the circle we should check for break points
        we check every interval*2*pi radians
        
    Returns:
    
        a (basis coefficients) and r (representation of rotations as a complex 
        number)
    """
    frames=w.shape[0]
    points=w.shape[2]
    basis=e.shape[0]
    check=np.arange(0,1,interval)*2*np.pi
    a=np.empty((check.size,basis,frames))
    residue=np.empty((check.size,frames))
    # call reestimate_a 100 times
    r=np.empty(2)
    if (Lambda.size!=0):
        res=np.zeros((frames,points*2+basis+points))
        proj_e=np.zeros((basis,2*points+basis+points))
        d=np.diag(Lambda)
    else:
        res=np.empty((frames,points*2+basis))
        proj_e=np.empty((basis,2*points+basis))
        
    for i in xrange(check.size):
        c=check[i]
        r[0]=np.sin(c)
        r[1]=np.cos(c)
        grot=camera_r.dot(upgrade_r(r).T)
        rot=grot[:2]
        Ps=rot.dot(s0)
        res[:,:points*2]=w.reshape((frames,points*2))
        res[:,:points*2]-=Ps.reshape(points*2)
        if (Lambda.size!=0):
            proj_e[:,2*points:2*points+basis]=d
            proj_e[:,2*points+basis:]=depth_reg*grot[2].dot(e)
            res[:,2*points:].fill(0)
        proj_e[:,:2*points]=rot.dot(e).transpose(1,0,2).reshape(e.shape[0],2*points)
        a[i], residue[i], _, _ = scipy.linalg.lstsq(proj_e.T, res.T,
                                                             overwrite_a=True,
                                                             overwrite_b=True)
    #find and return best coresponding solution
    best=np.argmin(residue,0)
    assert(best.shape[0]==frames)
    theta=check[best]
    a=a.transpose(0,2,1)[(best,np.arange(frames))]
    #residue=residue[index]
    r=np.empty((2,frames))
    r[0]=np.sin(theta)
    r[1]=np.cos(theta)    
    return a,r

def estimate_a_and_r_with_scale(np.ndarray[DTYPE_t, ndim=3] w,
                     np.ndarray[DTYPE_t, ndim=3] e,
                     np.ndarray[DTYPE_t, ndim=2] camera_r=np.asarray([[1,0,0],
                                                                      [0,0,-1],
                                                                      [0,1,0]]),
                     np.ndarray[DTYPE_t, ndim=1] Lambda=np.ones(0),
                     DTYPE_t interval=0.01,
                     DTYPE_t scale_prior=-0.0014,                     
                     DTYPE_t depth_reg=0):
    """ Now with depth regularisation.
    So local optima are a problem in general.
    However:
    
        1. This problem is convex in a but not in r, and
        
        2. each frame can be solved independently.
        
    So for each frame, we can do a grid search in r and take the globally 
    optimal solution.
    
    In practice, we just brute force over 100 different estimates of r, and take
    the best pair (r,a*(r)) where a*(r) is the optimal minimiser of a given r.
    
    Arguments:
    
        w is a 3d measurement matrix of form frames*2*points
        
        e is a 3d set of basis vectors of from basis*3*points
        
        s0 is the 3d rest shape of form 3*points
        
        Lambda are the regularisor coefficients on the coefficients of the weights
        typically generated using PPCA
        
        interval is how far round the circle we should check for break points
        we check every interval*2*pi radians
        
    Returns:
    
        a (basis coefficients) and r (representation of rotations as a complex 
        number)
    """
    frames=w.shape[0]
    points=w.shape[2]
    basis=e.shape[0]
    check=np.arange(0,1,interval)*2*np.pi
    a=np.empty((check.size,basis,frames))
    residue=np.empty((check.size,frames))
    # call reestimate_a 100 times
    r=np.empty(2)
    if (Lambda.size!=0):
        res=np.zeros((frames,points*2+basis+points))
        proj_e=np.zeros((basis,2*points+basis+points))
        d=np.diag(Lambda)
    else:
        res=np.empty((frames,points*2+basis))
        proj_e=np.empty((basis,2*points+basis))
        
    for i in xrange(check.size):
        c=check[i]
        r[0]=np.sin(c)
        r[1]=np.cos(c)
        grot=camera_r.dot(upgrade_r(r).T)
        rot=grot[:2]
        res[:,:points*2]=w.reshape((frames,points*2))
        if (Lambda.size!=0):
            proj_e[:,2*points:2*points+basis]=d
            proj_e[:,2*points+basis:]=depth_reg*grot[2].dot(e)
            res[:,2*points:].fill(0)
            res[:,2*points]=scale_prior #Keep to allow damping of scale
        proj_e[:,:2*points]=rot.dot(e).transpose(1,0,2).reshape(e.shape[0],2*points)
        a[i], residue[i], _, _ = scipy.linalg.lstsq(proj_e.T, res.T,
                                                    overwrite_a=True,
                                                    overwrite_b=True)
        
    #find and return best coresponding solution
    best=np.argmin(residue,0)
    assert(best.shape[0]==frames)
    theta=check[best]
    a=a.transpose(0,2,1)[(best,np.arange(frames))]
    #residue=residue[index]
    r=np.empty((2,frames))
    r[0]=np.sin(theta)
    r[1]=np.cos(theta)    
    return a,r
    
def estimate_a_and_r_with_signed_scale(np.ndarray[DTYPE_t, ndim=3] w,
                     np.ndarray[DTYPE_t, ndim=3] e,
                     np.ndarray[DTYPE_t, ndim=1] Lambda,
                     np.ndarray[DTYPE_t, ndim=3] weights,
                     np.ndarray[DTYPE_t, ndim=2] camera_r,
                     DTYPE_t interval=0.01,                     
                     DTYPE_t depth_reg=0,
                     DTYPE_t scale_prior=-0.0014,                     
                     ):
    """ Now with depth regularisation.
    So local optima are a problem in general.
    However:
    
        1. This problem is convex in a but not in r, and
        
        2. each frame can be solved independently.
        
    So for each frame, we can do a grid search in r and take the globally 
    optimal solution.
    
    In practice, we just brute force over 100 different estimates of r, and take
    the best pair (r,a*(r)) where a*(r) is the optimal minimiser of a given r.
    
    Arguments:
    
        w is a 3d measurement matrix of form frames*2*points
        
        e is a 3d set of basis vectors of from basis*3*points
        
        s0 is the 3d rest shape of form 3*points
        
        Lambda are the regularisor coefficients on the coefficients of the weights
        typically generated using PPCA
        
        interval is how far round the circle we should check for break points
        we check every interval*2*pi radians
        
    Returns:
    
        a (basis coefficients) and r (representation of rotations as a complex 
        number)
    """
        
    frames=w.shape[0]
    points=w.shape[2]
    basis=e.shape[0]
    check=np.arange(0,1,interval)*2*np.pi
    a=np.empty((check.size,basis,frames))
    residue=np.empty((check.size,frames))
    # call reestimate_a 100 times
    r=np.empty(2)
    upper=np.empty(basis)
    upper.fill(np.inf)
    upper[0]=scale_prior
    # call reestimate_a 100 times
    r=np.empty(2)
    res=np.zeros((frames,points*2+basis+points))
    proj_e=np.zeros((basis,2*points+basis+points))
    p=np.empty_like(proj_e)
    d=np.diag(Lambda)
    weights2=weights.reshape(frames,points*2)
    w2=w.reshape(frames,points*2)
    for i in xrange(check.size):
        c=check[i]
        r[0]=np.sin(c)
        r[1]=np.cos(c)
        grot=camera_r.dot(upgrade_r(r).T)
        rot=grot[:2]
        res[:,:points*2]=w2
        proj_e[:,2*points:2*points+basis]=d
        proj_e[:,2*points+basis:]=depth_reg*grot[2].dot(e)
        res[:,2*points:].fill(0)
        res[:,2*points]=scale_prior*Lambda[0] #Keep to allow damping of scale
    
        res[:,:points*2]*=weights2
        proj_e[:,:2*points]=rot.dot(e).transpose(1,0,2).reshape(e.shape[0],2*points)
        for j in xrange(frames):
            p[:]=proj_e
            p[:,:2*points]*=weights2[j]
            a[i,:,j], residue[i,j], _, _ = scipy.linalg.lstsq(p.T, res[j].T)
            if a[i,0,j]>scale_prior:
                if (i==0):  # There is a bug in cython relating to short circuiting
                            #with or
                            # see https://github.com/cython/cython/issues/967
                    out=op.lsq_linear(p.T,res[j],bounds=(-np.inf,upper))
                    a[i,:,j]=out.x
                    residue[i,j]=out.cost
    
                elif (residue [i,j]<np.min(residue[:i,j])):
                    out=op.lsq_linear(p.T,res[j],bounds=(-np.inf,upper))
                    a[i,:,j]=out.x
                    residue[i,j]=out.cost
    
    #find and return best coresponding solution
    best=np.argmin(residue,0)
    assert(best.shape[0]==frames)
    theta=check[best]
    a=a.transpose(0,2,1)[(best,np.arange(frames))]
    #residue=residue[index]
    r=np.empty((2,frames))
    r[0]=np.sin(theta)
    r[1]=np.cos(theta)    
    return a,r

def estimate_a_and_r_with_weights(np.ndarray[DTYPE_t, ndim=3] w,
                     np.ndarray[DTYPE_t, ndim=3] e,
                     np.ndarray[DTYPE_t, ndim=2] camera_r,
                     np.ndarray[DTYPE_t, ndim=1] Lambda,
                     np.ndarray[DTYPE_t, ndim=3] weights,
                     DTYPE_t scale=.0014,
                     DTYPE_t interval=0.01,                     
                     DTYPE_t depth_reg=0):
    """ Now with depth regularisation.
    So local optima are a problem in general.
    However:
    
        1. This problem is convex in a but not in r, and
        
        2. each frame can be solved independently.
        
    So for each frame, we can do a grid search in r and take the globally 
    optimal solution.
    
    In practice, we just brute force over 100 different estimates of r, and take
    the best pair (r,a*(r)) where a*(r) is the optimal minimiser of a given r.
    
    Arguments:
    
        w is a 3d measurement matrix of form frames*2*points
        
        e is a 3d set of basis vectors of from basis*3*points
        
        s0 is the 3d rest shape of form 3*points
        
        Lambda are the regularisor coefficients on the coefficients of the weights
        typically generated using PPCA
        
        interval is how far round the circle we should check for break points
        we check every interval*2*pi radians
        
    Returns:
    
        a (basis coefficients) and r (representation of rotations as a complex 
        number)
    """
    frames=w.shape[0]
    points=w.shape[2]
    basis=e.shape[0]
    check=np.arange(0,1,interval)*2*np.pi
    a=np.empty((check.size,basis,frames))
    residue=np.empty((check.size,frames))
    # call reestimate_a 100 times
    r=np.empty(2)
    res=np.zeros((frames,points*2+basis+points))
    proj_e=np.zeros((basis,2*points+basis+points))
    p=np.empty_like(proj_e)
    d=np.diag(Lambda)
    weights2=weights.reshape(frames,points*2)
    w2=w.reshape(frames,points*2)
    for i in xrange(check.size):
        c=check[i]
        r[0]=np.sin(c)
        r[1]=np.cos(c)
        grot=camera_r.dot(upgrade_r(r).T)
        rot=grot[:2]
        res[:,:points*2]=w2
        proj_e[:,2*points:2*points+basis]=d
        proj_e[:,2*points+basis:]=depth_reg*grot[2].dot(e)
        res[:,2*points:].fill(0)
        res[:,2*points]=scale #Keep to allow damping of scale
        res[:,:points*2]*=weights2
        proj_e[:,:2*points]=rot.dot(e).transpose(1,0,2).reshape(e.shape[0],2*points)
        for j in xrange(frames):
            p[:]=proj_e
            p[:,:2*points]*=weights2[j]
            a[i,:,j], residue[i,j], _, _ = scipy.linalg.lstsq(p.T, res[j].T,
                                                             overwrite_a=True,
                                                             overwrite_b=True)
    #find and return best coresponding solution
    best=np.argmin(residue,0)
    assert(best.shape[0]==frames)
    theta=check[best]
    a=a.transpose(0,2,1)[(best,np.arange(frames))]
    #residue=residue[index]
    r=np.empty((2,frames))
    r[0]=np.sin(theta)
    r[1]=np.cos(theta)    
    return a,r

def estimate_a_and_r_with_weights_fixed_scale(np.ndarray[DTYPE_t, ndim=3] w,
                     np.ndarray[DTYPE_t, ndim=3] e,
                     np.ndarray[DTYPE_t, ndim=2] mu,
                     np.ndarray[DTYPE_t, ndim=2] camera_r,
                     np.ndarray[DTYPE_t, ndim=1] Lambda,
                     np.ndarray[DTYPE_t, ndim=3] weights,
                     DTYPE_t interval=0.01,                     
                     DTYPE_t depth_reg=0):
    """ Now with depth regularisation.
    So local optima are a problem in general.
    However:
    
        1. This problem is convex in a but not in r, and
        
        2. each frame can be solved independently.
        
    So for each frame, we can do a grid search in r and take the globally 
    optimal solution.
    
    In practice, we just brute force over 100 different estimates of r, and take
    the best pair (r,a*(r)) where a*(r) is the optimal minimiser of a given r.
    
    Arguments:
    
        w is a 3d measurement matrix of form frames*2*points
        
        e is a 3d set of basis vectors of from basis*3*points
        
        s0 is the 3d rest shape of form 3*points
        
        Lambda are the regularisor coefficients on the coefficients of the weights
        typically generated using PPCA
        
        interval is how far round the circle we should check for break points
        we check every interval*2*pi radians
        
    Returns:
    
        a (basis coefficients) and r (representation of rotations as a complex 
        number)
    """
    frames=w.shape[0]
    points=w.shape[2]
    basis=e.shape[0]
    check=np.arange(0,1,interval)*2*np.pi
    a=np.empty((check.size,basis,frames))
    residue=np.empty((check.size,frames))
    # call reestimate_a 100 times
    r=np.empty(2)
    res=np.zeros((frames,points*2+basis+points))
    proj_e=np.zeros((basis,2*points+basis+points))
    proj_mu=np.empty((2,points))
    p=np.empty_like(proj_e)
    d=np.diag(Lambda)
    weights2=weights.reshape(frames,points*2)
    w2=w.reshape(frames,points*2)
    for i in xrange(check.size):
        c=check[i]
        r[0]=np.sin(c)
        r[1]=np.cos(c)
        grot=camera_r.dot(upgrade_r(r).T)
        rot=grot[:2]
        res[:,:points*2]=w2
        res[:,:points*2]-=rot.dot(mu,proj_mu).reshape(points*2)
        proj_e[:,2*points:2*points+basis]=d
        proj_e[:,2*points+basis:]=depth_reg*grot[2].dot(e)
        res[:,2*points:].fill(0)
        res[:,:points*2]*=weights2
        proj_e[:,:2*points]=rot.dot(e).transpose(1,0,2).reshape(e.shape[0],2*points)
        for j in xrange(frames):
            p[:]=proj_e
            p[:,:2*points]*=weights2[j]
            a[i,:,j], residue[i,j], _, _ = scipy.linalg.lstsq(p.T, res[j].T,
                                                             overwrite_a=True,
                                                             overwrite_b=True)
    #find and return best coresponding solution
    best=np.argmin(residue,0)
    assert(best.shape[0]==frames)
    theta=check[best]
    a=a.transpose(0,2,1)[(best,np.arange(frames))]
    #residue=residue[index]
    r=np.empty((2,frames))
    r[0]=np.sin(theta)
    r[1]=np.cos(theta)    
    return a,r


def estimate_a_and_r_soft_max(np.ndarray[DTYPE_t, ndim=3] w,
                              np.ndarray[DTYPE_t, ndim=3] e,
                         np.ndarray[DTYPE_t, ndim=2] s0,
                         np.ndarray[DTYPE_t, ndim=2] camera_r=np.asarray([[1,0,0],
                                                                         [0,0,-1],
                                                                         [0,1,0]]),
                         np.ndarray[DTYPE_t, ndim=1] Lambda=np.ones(0),
                         np.ndarray[DTYPE_t, ndim=3] weights=np.ones((0,0,0)),
                         DTYPE_t interval=0.01):
    """So local optima are a problem in general.
    However:
    
        1. This problem is convex in a but not in r, and
        
        2. each frame can be solved independently.
        
    So for each frame, we can do a grid search in r and take the globally 
    optimal solution.
    
    In practice, we just brute force over 100 different estimates of r, and take
    the best pair (r,a*(r)) where a*(r) is the optimal minimiser of a given r.
    
    Arguments:
    
        w is a 3d measurement matrix of form frames*2*points
        
        e is a 3d set of basis vectors of from basis*3*points
        
        s0 is the 3d rest shape of form 3*points
        
        Lambda are the regularisor coefficients on the coefficients of the weights
        typically generated using PPCA
        
        interval is how far round the circle we should check for break points
        we check every interval*2*pi radians
        
    Returns:
    
        a (basis coefficients) and r (representation of rotations as a complex 
        number)
    """
    frames=w.shape[0]
    points=w.shape[2]
    basis=e.shape[0]
    check=np.arange(0,1,interval)*2*np.pi
    a=np.empty((check.size,basis,frames))
    residue=np.empty((check.size,frames))
    # call reestimate_a 100 times
    r=np.empty(2)
    if (Lambda.size!=0):
        res=np.zeros((frames,points*2+basis))
        proj_e=np.zeros((basis,2*points+basis))
        d=np.diag(Lambda)
    else:
        res=np.empty((frames,points*2))
        proj_e=np.empty((basis,2*points))
        
    for i in xrange(check.size):
        c=check[i]
        r[0]=np.sin(c)
        r[1]=np.cos(c)
        rot=camera_r[:2].dot(upgrade_r(r).T)
        Ps=rot.dot(s0)
        res[:,:points*2]=w.reshape((frames,points*2))
        res[:,:points*2]-=Ps.reshape(points*2)
        if (Lambda.size!=0):
            proj_e[:,2*points:]=d
            res[:,2*points:].fill(0)
        if weights.size!=0: 
            res[:,:points*2]*=weights
        proj_e[:,:2*points]=rot.dot(e).transpose(1,0,2).reshape(e.shape[0],2*points)
        a[i], residue[i], _, _ = scipy.linalg.lstsq(proj_e.T, res.T,
                                                             overwrite_a=True,
                                                             overwrite_b=True)
    #find and return best coresponding solution
    best=np.min(residue,0)
    residue-=best
    return a,residue,check
    
cdef estimate_a_and_r_with_res(np.ndarray[DTYPE_t, ndim=3] w,
                     np.ndarray[DTYPE_t, ndim=3] e,
                     np.ndarray[DTYPE_t, ndim=2] s0,
                     np.ndarray[DTYPE_t, ndim=2] camera_r,
                     np.ndarray[DTYPE_t, ndim=1] Lambda,
                     np.ndarray[DTYPE_t, ndim=1] check,
                     np.ndarray[DTYPE_t, ndim=3] a,
                     np.ndarray[DTYPE_t, ndim=3] weights,
                     np.ndarray[DTYPE_t, ndim=2] res,
                     np.ndarray[DTYPE_t, ndim=2] proj_e,
                     np.ndarray[DTYPE_t, ndim=2] residue,
                     np.ndarray[DTYPE_t, ndim=2] Ps,
                     DTYPE_t depth_reg,
                     DTYPE_t scale_prior
                     ):
    """So local optima are a problem in general.
    However:
    
        1. This problem is convex in a but not in r, and
        
        2. each frame can be solved independently.
        
    So for each frame, we can do a grid search in r and take the globally 
    optimal solution.
    
    In practice, we just brute force over 100 different estimates of r, and take
    the best pair (r,a*(r)) where a*(r) is the optimal minimiser of a given r.
    
    Arguments:
    
        w is a 3d measurement matrix of form frames*2*points
        
        e is a 3d set of basis vectors of from basis*3*points
        
        s0 is the 3d rest shape of form 3*points
        
        Lambda are the regularisor coefficients on the coefficients of the weights
        typically generated using PPCA
        
        interval is how far round the circle we should check for break points
        we check every interval*2*pi radians
        
    Returns:
    
        a (basis coefficients) and r (representation of rotations as a complex 
        number)
    """
    frames=w.shape[0]
    points=w.shape[2]
    basis=e.shape[0]
    r=np.empty(2)
    Ps_reshape=Ps.reshape(2*points)
    w_reshape=w.reshape((frames,points*2))  
    if (Lambda.size!=0):
         d=np.diag(Lambda[:Lambda.shape[0]-1])
   
    for i in xrange(check.size):
        c=check[i]
        r[0]=np.sin(c)
        r[1]=np.cos(c)
        grot=camera_r.dot(upgrade_r(r).T)
        rot=grot[:2]
        rot.dot(s0,Ps)
        res[:,:points*2]=w_reshape
        res[:,:points*2]-=Ps_reshape
        proj_e[:,:2*points]=rot.dot(e).transpose(1,0,2).reshape(e.shape[0],2*points)

        if (Lambda.size!=0):
            proj_e[:,2*points:2*points+basis]=d#/Lambda[Lambda.shape[0]-1])
            res[:,2*points:].fill(0)
            res[:,:points*2]*=Lambda[Lambda.shape[0]-1]
            proj_e[:,:points*2]*=Lambda[Lambda.shape[0]-1]
            proj_e[:,2*points+basis:]=((Lambda[Lambda.shape[0]-1] * 
                                        depth_reg)*grot[2]).dot(e)
            res[:,2*points:].fill(0)
            res[:,2*points]=scale_prior
        if weights.size!=0: 
            res[:,:points*2]*=weights
            proj_e[:,:points*2]*=weights        
        a[i], residue[i], _, _ = scipy.linalg.lstsq(proj_e.T, res.T,
                                                             overwrite_a=True,
                                                             overwrite_b=True)
    #find and return best coresponding solution
    best=np.argmin(residue,0)
    assert(best.shape[0]==frames)
    theta=check[best]
    index=(best,np.arange(frames))
    aa=a.transpose(0,2,1)[index]
    retres=residue[index]
    r=np.empty((2,frames))
    r[0]=np.sin(theta)
    r[1]=np.cos(theta)    
    return aa,r,retres    
        
cdef estimate_a_and_r_with_res_weights(np.ndarray[DTYPE_t, ndim=3] w,
                     np.ndarray[DTYPE_t, ndim=3] e,
                     np.ndarray[DTYPE_t, ndim=2] s0,
                     np.ndarray[DTYPE_t, ndim=2] camera_r,
                     np.ndarray[DTYPE_t, ndim=1] Lambda,
                     np.ndarray[DTYPE_t, ndim=1] check,
                     np.ndarray[DTYPE_t, ndim=3] a,
                     np.ndarray[DTYPE_t, ndim=2] weights,
                     np.ndarray[DTYPE_t, ndim=2] res,
                     np.ndarray[DTYPE_t, ndim=2] proj_e,
                     np.ndarray[DTYPE_t, ndim=2] residue,
                     np.ndarray[DTYPE_t, ndim=2] Ps,
                     DTYPE_t depth_reg,
                     DTYPE_t scale_prior
                     ):
    """So local optima are a problem in general.
    However:
    
        1. This problem is convex in a but not in r, and
        
        2. each frame can be solved independently.
        
    So for each frame, we can do a grid search in r and take the globally 
    optimal solution.
    
    In practice, we just brute force over 100 different estimates of r, and take
    the best pair (r,a*(r)) where a*(r) is the optimal minimiser of a given r.
    
    Arguments:
    
        w is a 3d measurement matrix of form frames*2*points
        
        e is a 3d set of basis vectors of from basis*3*points
        
        s0 is the 3d rest shape of form 3*points
        
        Lambda are the regularisor coefficients on the coefficients of the weights
        typically generated using PPCA
        
        interval is how far round the circle we should check for break points
        we check every interval*2*pi radians
        
    Returns:
    
        a (basis coefficients) and r (representation of rotations as a complex 
        number)
    """
    frames=w.shape[0]
    points=w.shape[2]
    basis=e.shape[0]
    r=np.empty(2)
    Ps_reshape=Ps.reshape(2*points)
    w_reshape=w.reshape((frames,points*2))
    p_copy=np.empty_like(proj_e)
    if (Lambda.size!=0):
         d=np.diag(Lambda[:Lambda.shape[0]-1])
   
    for i in xrange(check.size):
        c=check[i]
        r[0]=np.sin(c)
        r[1]=np.cos(c)
        grot=camera_r.dot(upgrade_r(r).T)
        rot=grot[:2]
        rot.dot(s0,Ps)
        res[:,:points*2]=w_reshape
        res[:,:points*2]-=Ps_reshape
        proj_e[:,:2*points]=rot.dot(e).transpose(1,0,2).reshape(e.shape[0],2*points)

        if (Lambda.size!=0):
            proj_e[:,2*points:2*points+basis]=d#/Lambda[Lambda.shape[0]-1])
            res[:,2*points:].fill(0)
            res[:,:points*2]*=Lambda[Lambda.shape[0]-1]
            proj_e[:,:points*2]*=Lambda[Lambda.shape[0]-1]
            proj_e[:,2*points+basis:]=((Lambda[Lambda.shape[0]-1] * 
                                        depth_reg)*grot[2]).dot(e)
            res[:,2*points:].fill(0)
            res[:,2*points]=scale_prior
        if weights.size!=0: 
            res[:,:points*2]*=weights
        for j in xrange(frames):
            p_copy[:]=proj_e
            p_copy[:,:points*2]*=weights[j]    
            a[i,:,j], residue[i,j], _, _ = np.linalg.lstsq(p_copy.T, res[j].T)
    #find and return best coresponding solution
    best=np.argmin(residue,0)
    assert(best.shape[0]==frames)
    theta=check[best]
    index=(best,np.arange(frames))
    aa=a.transpose(0,2,1)[index]
    retres=residue[index]
    r=np.empty((2,frames))
    r[0]=np.sin(theta)
    r[1]=np.cos(theta)    
    return aa,r,retres    
    
cdef upgrade_r(np.ndarray[DTYPE_t, ndim=1] r):
    """Upgrades complex parameterisation of planar rotation to tensor containing
    per frame 3x3 rotation matrices"""
    newr = np.zeros((3, 3))
    newr[:2, 0] = r
    newr[2, 2] = 1
    newr[1::-1, 1] = r
    newr[0, 1] *= -1
    return newr

cdef upgrade_r_full(r):
    """Upgrades complex parameterisation of planar rotation to tensor containing
    per frame 3x3 rotation matrices"""
    assert(r.ndim==2)
    #print r.shape
    #assert(r.shape[1]==2) # Technically optional assert, but if this fails 
                          # Data is probably transposed
    #assert(np.all(np.isfinite(r)))
    #norm=np.sqrt((r[:,:2]**2).sum(1))
    #assert(np.all(norm>0))
    #r/=norm[:,np.newaxis]
    #assert(np.all(np.isfinite(r)))
    newr = np.zeros((r.shape[0], 3, 3))
    newr[:, :2, 0] = r[:, :2]
    #newr[:, :2, 0] *= norm[:,np.newaxis]
    newr[:, 2, 2] = 1
    newr[:, 1::-1, 1] = r[:, :2]
    #newr[:, 1::-1, 1] *= norm[:,np.newaxis]
    newr[:,0, 1] *= -1
    return newr
