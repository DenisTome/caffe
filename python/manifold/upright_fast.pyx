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
    print w_total.shape[0],w_total.shape[1],w_total.shape[2]
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

#def reestimate_r_alt_camera(np.ndarray[DTYPE_t, ndim=2] w,
#                             np.ndarray[DTYPE_t, ndim=2] shape,
#                             np.ndarray[DTYPE_t, ndim=2] a,
#                             np.ndarray[DTYPE_t, ndim=3] e,
#                             r_init=False,
#                             camera_r=np.identity(3)):
#     """Reestimate r from rest shape, basis and basis coefficients
#     with new camera parameters (rotation)"""
#     if r_init is not False:
#         assert(r_init.shape[0]==2)
#         assert(r_init.shape[1]==a.shape[0])
#     #to do, computer xzdat from R
#     d = a.dot(exz.reshape(a.shape[1], -1)).reshape(a.shape[0], 3, -1)
#     d += shape
#     if r_init is not False:
#         r_init=np.arccos(r_init[1])*np.sign(r_init[0])
#
#    r=get_rotation_funky_offset_c(w,d,r_init,camera_r)
#    rot=np.empty((2,xzdat.shape[0]))
#    np.sin(r,rot[0])
#    np.cos(r,rot[1])
#    return rot
