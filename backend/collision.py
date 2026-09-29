"""Mesh-derived capsule self-collision cleanup with articulated joint projection."""
import numpy as np
from scipy.spatial.transform import Rotation
from timeline import forward_kinematics, to_local

def closest(a,b,c,d):
    """Exact closest points for batches of segments, including sphere capsules."""
    u,v=b-a,d-c
    uu,vv=np.einsum("ij,ij->i",u,u),np.einsum("ij,ij->i",v,v)
    uv=np.einsum("ij,ij->i",u,v)
    w=a-c
    uw,vw=np.einsum("ij,ij->i",u,w),np.einsum("ij,ij->i",v,w)
    denom=uu*vv-uv*uv
    ss=(uv*vw-vv*uw)/np.maximum(denom,1e-20)
    tt=(uu*vw-uv*uw)/np.maximum(denom,1e-20)
    valid=(denom>1e-15)&(ss>=0)&(ss<=1)&(tt>=0)&(tt<=1)
    sa=np.stack([np.zeros_like(uu),np.ones_like(uu),
                 np.clip(-uw/np.maximum(uu,1e-20),0,1),
                 np.clip((uv-uw)/np.maximum(uu,1e-20),0,1),
                 np.clip(ss,0,1)])
    tb=np.stack([np.clip(vw/np.maximum(vv,1e-20),0,1),
                 np.clip((vw+uv)/np.maximum(vv,1e-20),0,1),
                 np.zeros_like(uu),np.ones_like(uu),np.clip(tt,0,1)])
    pa=a[None]+sa[...,None]*u[None]
    pb=c[None]+tb[...,None]*v[None]
    dist=np.linalg.norm(pa-pb,axis=-1)
    dist[-1,~valid]=np.inf
    k=dist.argmin(axis=0)
    return pa[k,np.arange(len(a))],pb[k,np.arange(len(a))]

class CapsuleRig:
    def __init__(self,skeleton):
        self.skeleton=skeleton
        self.parents=skeleton["parents"]
        self.heads=np.asarray(skeleton["heads"])
        caps=skeleton.get("collision_capsules",[])
        self.joints=np.array([c["joint"] for c in caps],dtype=int)
        self.a=np.array([c["a"] for c in caps])-self.heads[self.joints]
        self.b=np.array([c["b"] for c in caps])-self.heads[self.joints]
        self.radii=np.array([c["radius"] for c in caps])
        self.ancestors=[]
        for j in range(len(self.parents)):
            chain=[]
            while j>=0:
                chain.append(j)
                j=self.parents[j]
            self.ancestors.append(chain)
        pairs=[]
        chains=[]
        for i in range(len(caps)):
            for j in range(i+1,len(caps)):
                ac,bc=self.ancestors[self.joints[i]],self.ancestors[self.joints[j]]
                lca=next(k for k in ac if k in bc)
                # Neighbouring tissue belongs to a continuous skin surface.
                if ac.index(lca)+bc.index(lca)<=2:
                    continue
                pairs.append((i,j))
                chains.append((ac[:min(ac.index(lca),3)],bc[:min(bc.index(lca),3)]))
        self.pairs=np.array(pairs,dtype=int).reshape(-1,2)
        self.chains=chains
        self.clearance=self.radii[self.pairs].sum(axis=1) if pairs else np.array([])
        volume=np.zeros(len(self.parents))
        for c,j in enumerate(self.joints):
            mass=self.radii[c]**2*(np.linalg.norm(self.b[c]-self.a[c])+4*self.radii[c]/3)
            for parent in self.ancestors[j]:
                volume[parent]+=mass
        positive=volume[volume>0]
        scale=np.median(positive) if len(positive) else 1.
        self.mobility=np.clip(scale/np.maximum(volume,scale*.1),.03,3.)
        self.mobility[0]=0
        self.size=float(np.linalg.norm(np.ptp(self.heads,axis=0)))
        self.tolerance=self.size*.0005

    def world(self,pos,rot):
        a=pos[self.joints]+np.einsum("nij,nj->ni",rot[self.joints],self.a)
        b=pos[self.joints]+np.einsum("nij,nj->ni",rot[self.joints],self.b)
        return a,b

    def contacts(self,pos,rot):
        a,b=self.world(pos,rot)
        ia,ib=self.pairs.T
        p,q=closest(a[ia],b[ia],a[ib],b[ib])
        delta=p-q
        distance=np.linalg.norm(delta,axis=-1)
        return p,q,delta,distance

    def penetration(self,pos,rot):
        if not len(self.pairs):
            return 0.
        return float(np.maximum(0,self.clearance-self.contacts(pos,rot)[-1]).max(initial=0))

def solve(positions,rotations,skeleton,iterations=32):
    if not skeleton.get("collision_capsules"):
        return positions, rotations, dict(method="none", reason="No weighted mesh collision shapes")
    rig=CapsuleRig(skeleton)
    if not len(rig.pairs):
        return positions,rotations,dict(method="mesh capsules",pairs=0)
    local=to_local(rotations,skeleton["parents"])
    output_p,output_r=positions.copy(),rotations.copy()
    prior_normal=np.zeros((len(rig.pairs),3))
    correction=np.tile(np.eye(3),(len(rig.parents),1,1))
    before,after=[],[]
    active_counts=[]
    for t in range(len(positions)):
        before.append(rig.penetration(positions[t],rotations[t]))
        # Keep corrections coherent across time instead of popping between two
        # equally close sides of a thigh or body surface.
        carry=Rotation.from_rotvec(Rotation.from_matrix(correction).as_rotvec()*.96).as_matrix()
        pose_local=carry@local[t]
        p,r=forward_kinematics(positions[t:t+1,0],pose_local[None],skeleton)
        pos,rot=p[0],r[0]
        for iteration in range(iterations):
            pa,pb,delta,dist=rig.contacts(pos,rot)
            normal=delta/np.maximum(dist[:,None],1e-9)
            coincident=dist<1e-8
            normal[coincident]=np.array([0.,0.,1.])
            old=np.linalg.norm(prior_normal,axis=1)>.5
            near=dist<rig.clearance*1.35
            # Retain which side the limb approached from until it separates.
            keep=old&near
            alignment=np.sum(normal*prior_normal,axis=1)
            normal[keep&(alignment<.5)]=prior_normal[keep&(alignment<.5)]
            penetration=rig.clearance-np.sum(delta*normal,axis=1)
            active=np.flatnonzero(penetration>rig.tolerance)
            if not len(active):
                break
            updates=np.zeros((len(rig.parents),3))
            for pair in active:
                jacobians=[]
                for chain,point,sign in ((rig.chains[pair][0],pa[pair],1.),
                                          (rig.chains[pair][1],pb[pair],-1.)):
                    for joint in chain:
                        jac=sign*np.cross(point-pos[joint],normal[pair])
                        jacobians.append((joint,jac))
                denom=sum(rig.mobility[j]*np.dot(jac,jac) for j,jac in jacobians)
                if denom<1e-10:
                    continue
                for joint,jac in jacobians:
                    updates[joint]+=jac*rig.mobility[joint]*penetration[pair]/denom*.65
            lengths=np.linalg.norm(updates,axis=1,keepdims=True)
            updates*=np.minimum(1.,np.radians(4)/np.maximum(lengths,1e-12))
            for j,parent in enumerate(rig.parents):
                if parent>=0:
                    world_change=Rotation.from_rotvec(updates[j]).as_matrix()
                    pose_local[j]=(rot[parent].T@world_change@rot[parent])@pose_local[j]
            p,r=forward_kinematics(positions[t:t+1,0],pose_local[None],skeleton)
            pos,rot=p[0],r[0]
        pa,pb,delta,dist=rig.contacts(pos,rot)
        near=dist<rig.clearance*1.15
        current=delta/np.maximum(dist[:,None],1e-9)
        good=near&(np.sum(current*prior_normal,axis=1)>.5)
        prior_normal[good]=current[good]
        new=near&(np.linalg.norm(prior_normal,axis=1)<.5)
        prior_normal[new]=current[new]
        prior_normal[~near]=0
        correction=pose_local@local[t].swapaxes(-1,-2)
        output_p[t],output_r[t]=pos,rot
        after.append(rig.penetration(pos,rot))
        active_counts.append(int(np.sum(rig.clearance-dist>rig.tolerance)))
    report=dict(method="mesh-derived capsules with articulated projection",pairs=len(rig.pairs),
                max_penetration_before=max(before),max_penetration_after=max(after),
                tolerance=rig.tolerance,frames_with_remaining_contacts=sum(c>0 for c in active_counts),
                per_frame_before=before,per_frame_after=after)
    return output_p,output_r,report


def stabilize_feet(positions, rotations, skeleton, max_tilt=30.):
    """Limit excessive toe-up/down pitch, preserving heading and limb lengths."""
    local=to_local(rotations,skeleton["parents"])
    pos,rot=positions.copy(),rotations.copy()
    changed=[]
    for j,name in enumerate(skeleton["bone_names"]):
        if not name or not any(word in skeleton["labels"][j].split() for word in ("paw","foot")):
            continue
        parent=skeleton["parents"][j]
        if parent<0:
            continue
        rest=np.asarray(skeleton["rest_matrices"][j])[:3,1]
        direction=np.einsum("tij,j->ti",rot[:,j],rest)
        horizontal=direction[:,:2]
        length=np.linalg.norm(horizontal,axis=1)
        valid=length>1e-6
        heading=np.zeros_like(horizontal)
        heading[valid]=horizontal[valid]/length[valid,None]
        fallback=rest[:2]/max(np.linalg.norm(rest[:2]),1e-8)
        heading[~valid]=fallback
        angle=np.arctan2(direction[:,2],length)
        limit=np.radians(max_tilt)
        # Smooth saturation avoids a velocity kink at an angular clamp.
        desired_angle=limit*np.tanh(angle/limit)
        target=np.column_stack([heading*np.cos(desired_angle)[:,None],np.sin(desired_angle)])
        axes=np.cross(direction,target)
        sine=np.linalg.norm(axes,axis=1)
        cosine=np.clip(np.sum(direction*target,axis=1),-1.,1.)
        axes*=np.arctan2(sine,cosine)[:,None]/np.maximum(sine[:,None],1e-12)
        desired=Rotation.from_rotvec(axes).as_matrix()@rot[:,j]
        local[:,j]=rot[:,parent].swapaxes(-1,-2)@desired
        pos,rot=forward_kinematics(pos[:,0],local,skeleton)
        changed.append(dict(bone=name,maximum_tilt_before=float(np.degrees(abs(angle)).max()),
                            maximum_tilt_after=float(np.degrees(abs(desired_angle)).max())))
    return pos,rot,changed


def cleanup(positions,rotations,skeleton,ground=None):
    from ground import plant
    positions,rotations,initial=solve(positions,rotations,skeleton)
    positions,rotations,contact=plant(positions,rotations,skeleton,ground)
    positions,rotations,report=solve(positions,rotations,skeleton)
    report["initial_collision_pass"]=dict(
        maximum_overlap_before=initial.get("max_penetration_before",0.),
        maximum_overlap_after=initial.get("max_penetration_after",0.))
    report["ground_contact"]=contact
    report["foot_tilt_limits"]=contact.get("foot_limits",[])
    return positions,rotations,report
