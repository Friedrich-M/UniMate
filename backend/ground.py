"""Ground sampling, stance detection and contact-preserving limb correction."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, uniform_filter1d
from scipy.spatial.transform import Rotation
from timeline import forward_kinematics, to_local

def unit(value):
    return value / max(float(np.linalg.norm(value)), 1e-12)

def skew(value):
    x, y, z = value
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

class Surface:
    def __init__(self, ground, skeleton):
        self.normal = unit(np.asarray(ground.get("normal", [0.,0.,1.]),dtype=float))
        self.height = float(ground.get("height", 0.))
        helper = np.eye(3)[np.argmin(abs(self.normal))]
        self.u = unit(np.cross(self.normal, helper))
        self.v = np.cross(self.normal, self.u)
        tri = np.asarray(ground.get("triangles", []),dtype=float).reshape(-1,3,3)
        self.triangles = tri
        self.xy = np.stack([tri@self.u,tri@self.v],axis=-1) if len(tri) else np.empty((0,3,2))
        self.z = tri@self.normal if len(tri) else np.empty((0,3))
        self.face_normals = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]) if len(tri) else np.empty((0,3))
        sizes = np.linalg.norm(self.face_normals,axis=1)
        self.face_normals /= np.maximum(sizes[:,None],1e-12)
        self.face_normals *= np.sign(self.face_normals@self.normal)[:,None]
        self.valid = (sizes>1e-10)&((self.face_normals@self.normal)>.2)
        self.misses = 0

    def sample(self, point):
        if not len(self.triangles):
            return self.height,self.normal,True
        xy=np.array([np.dot(point,self.u),np.dot(point,self.v)])
        tri=self.xy
        a,b,c=tri[:,0],tri[:,1],tri[:,2]
        ab,ac,p=b-a,c-a,xy-a
        denominator=ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]
        safe=np.where(abs(denominator)>1e-12,denominator,1.)
        x=(p[:,0]*ac[:,1]-p[:,1]*ac[:,0])/safe
        y=(ab[:,0]*p[:,1]-ab[:,1]*p[:,0])/safe
        eligible=self.valid&(abs(denominator)>1e-10)&(x>=-1e-6)&(y>=-1e-6)&(x+y<=1+1e-6)
        if not eligible.any():
            self.misses+=1
            return self.height,self.normal,False
        levels=self.z[:,0]+x*(self.z[:,1]-self.z[:,0])+y*(self.z[:,2]-self.z[:,0])
        near=eligible&(levels<=np.dot(point,self.normal)+1.)
        if near.any():
            eligible=near
        index=int(np.argmax(np.where(eligible,levels,-np.inf)))
        return float(levels[index]),self.face_normals[index],True

def profiles_for(skeleton):
    if "foot_profiles" in skeleton:
        return skeleton["foot_profiles"]
    profiles=[]
    heads=np.asarray(skeleton["heads"])
    for j,name in enumerate(skeleton["bone_names"]):
        if not name or not ({"foot","paw"}&set(skeleton["labels"][j].split())):
            continue
        p=skeleton["parents"][j]
        if p<0 or skeleton["parents"][p]<0:
            continue
        upper=skeleton["parents"][p]
        rest=np.asarray(skeleton["rest_matrices"][j])[:3,1]
        pitch=np.degrees(np.arctan2(rest[2],np.linalg.norm(rest[:2])))
        stance=float(np.clip(abs(pitch)+15,20,45))
        profiles.append(dict(joint=j,parent=p,upper=upper,
            leg_length=float(np.linalg.norm(heads[j]-heads[p])+np.linalg.norm(heads[p]-heads[upper])),
            stance_tilt=stance,swing_tilt=max(stance+15,45)))
    return profiles

def foot_capsules(skeleton,profiles):
    caps={c["joint"]:c for c in skeleton.get("collision_capsules",[])}
    heads=np.asarray(skeleton["heads"])
    found=[]
    for profile in profiles:
        j=profile["joint"]
        if j not in caps:
            continue
        c=caps[j]
        found.append(dict(profile=profile,offset=np.asarray(c["b"])-heads[j],radius=float(c["radius"])))
    return found

def sole_points(pos,rot,foot,surface):
    j=foot["profile"]["joint"]
    center=pos[:,j]+np.einsum("tij,j->ti",rot[:,j],foot["offset"])
    return center-surface.normal*foot["radius"]

def foot_metrics(positions,rotations,feet,surface):
    all_metrics=[]
    for foot in feet:
        soles=sole_points(positions,rotations,foot,surface)
        height=np.empty(len(soles))
        normals=np.empty_like(soles)
        for t,sole in enumerate(soles):
            level,normal,_=surface.sample(sole)
            height[t]=np.dot(sole,surface.normal)-level
            normals[t]=normal
        all_metrics.append(dict(soles=soles,height=height,normals=normals))
    return all_metrics

def runs(mask,min_length=4):
    result=[]
    start=None
    for i,value in enumerate(np.r_[mask,False]):
        if value and start is None:
            start=i
        elif not value and start is not None:
            if i-start>=min_length:
                result.append((start,i))
            start=None
    return result

def stance_windows(positions,rotations,feet,surface):
    raw=foot_metrics(positions,rotations,feet,surface)
    windows=[]
    for item,foot in zip(raw,feet):
        leg=foot["profile"]["leg_length"]
        sole=item["soles"]
        raw_velocity=np.linalg.norm(np.diff(sole,axis=0,prepend=sole[:1]),axis=1)
        velocity=uniform_filter1d(raw_velocity,size=3,mode="nearest")
        height=uniform_filter1d(item["height"],size=3,mode="nearest")
        vertical=np.abs(np.diff(item["height"],prepend=item["height"][:1]))
        near=height < max(.025,.16*leg)
        slow=(velocity < .08*leg)&(raw_velocity < .10*leg)&(vertical < .08*leg)
        windows.append(runs(near&slow))
    return windows

def soft_tilt(angle,limit):
    shoulder=.6*limit
    extra=np.maximum(abs(angle)-shoulder,0)
    return np.sign(angle)*(np.minimum(abs(angle),shoulder)+
           (limit-shoulder)*np.tanh(extra/np.maximum(limit-shoulder,1e-8)))

def stabilize_feet(positions,rotations,skeleton,feet,surface,windows):
    local=to_local(rotations,skeleton["parents"])
    pos,rot=positions.copy(),rotations.copy()
    changed=[]
    for foot,segments in zip(feet,windows):
        j=foot["profile"]["joint"]
        parent=skeleton["parents"][j]
        rest=np.asarray(skeleton["rest_matrices"][j])[:3,1]
        direction=np.einsum("tij,j->ti",rot[:,j],rest)
        dot=direction@surface.normal
        horizontal=direction-dot[:,None]*surface.normal
        length=np.linalg.norm(horizontal,axis=1)
        heading=horizontal/np.maximum(length[:,None],1e-12)
        fallback=rest-surface.normal*np.dot(rest,surface.normal)
        right=np.asarray(skeleton["rest_matrices"][j])[:3,0]
        side=np.einsum("tij,j->ti",rot[:,j],right)
        alternative=np.cross(surface.normal,side)
        alternative/=np.maximum(np.linalg.norm(alternative,axis=1)[:,None],1e-12)
        sign=np.sign(np.dot(np.cross(surface.normal,right),fallback))
        alternative*=sign if sign else 1.
        # Toe heading is ambiguous when the foot points almost vertically.
        weight=np.clip((length-.15)/.25,0,1)[:,None]
        heading=heading*weight+alternative*(1-weight)
        heading/=np.maximum(np.linalg.norm(heading,axis=1)[:,None],1e-12)
        heading[length<1e-7]=unit(fallback)
        angle=np.arctan2(dot,length)
        weight=np.zeros(len(pos))
        for start,end in segments:
            for t in range(start,end):
                weight[t]=min(1.,(t-start+1)/4,(end-t)/4)
        stance=np.radians(foot["profile"]["stance_tilt"])
        swing=np.radians(foot["profile"]["swing_tilt"])
        limit=swing+(stance-swing)*weight
        desired_angle=soft_tilt(angle,limit)
        target=heading*np.cos(desired_angle)[:,None]+surface.normal*np.sin(desired_angle)[:,None]
        axes=np.cross(direction,target)
        sine=np.linalg.norm(axes,axis=1)
        cosine=np.clip(np.sum(direction*target,axis=1),-1.,1.)
        axes*=np.arctan2(sine,cosine)[:,None]/np.maximum(sine[:,None],1e-12)
        desired=Rotation.from_rotvec(axes).as_matrix()@rot[:,j]
        # Avoid yaw flips when a near-vertical toe changes its horizontal sign.
        original=rot[:,j].copy()
        for t in range(1,len(desired)):
            raw=Rotation.from_matrix(original[t-1].T@original[t]).magnitude()
            cap=max(np.radians(8),raw+np.radians(5))
            step=Rotation.from_matrix(desired[t-1].T@desired[t]).as_rotvec()
            step_angle=np.linalg.norm(step)
            if step_angle>cap:
                desired[t]=desired[t-1]@Rotation.from_rotvec(step*cap/step_angle).as_matrix()
        local[:,j]=rot[:,parent].swapaxes(-1,-2)@desired
        pos,rot=forward_kinematics(pos[:,0],local,skeleton)
        changed.append(dict(bone=skeleton["bone_names"][j],
                            stance_limit=foot["profile"]["stance_tilt"],
                            swing_limit=foot["profile"]["swing_tilt"],
                            maximum_tilt_before=float(np.degrees(abs(angle)).max()),
                            maximum_tilt_after=float(np.degrees(abs(desired_angle)).max())))
    return pos,rot,changed

def anchors_for(positions,rotations,feet,surface,windows):
    metrics=foot_metrics(positions,rotations,feet,surface)
    anchors={}
    for index,(foot,segments) in enumerate(zip(feet,windows)):
        sole=metrics[index]["soles"]
        for start,end in segments:
            base=sole[start].copy()
            level,normal,hit=surface.sample(base)
            target=base+surface.normal*(level-np.dot(base,surface.normal))
            anchors[(index,start,end)]=dict(point=target,normal=normal,hit=hit)
    return anchors

def contact_point(pos,rot,foot,surface):
    j=foot["profile"]["joint"]
    return pos[j]+rot[j]@foot["offset"]-surface.normal*foot["radius"]

def solve_contacts(positions,rotations,skeleton,feet,surface,windows):
    anchors=anchors_for(positions,rotations,feet,surface,windows)
    local=to_local(rotations,skeleton["parents"])
    output_p,output_r=positions.copy(),rotations.copy()
    start_points=positions.copy()
    previous_correction=np.tile(np.eye(3),(len(skeleton["parents"]),1,1))
    previous_root_shift=np.zeros(3)
    active_frames=0
    for t in range(len(positions)):
        constraints=[]
        for i,segments in enumerate(windows):
            for start,end in segments:
                if start<=t<end:
                    leg=feet[i]["profile"]["leg_length"]
                    anchor=anchors[(i,start,end)]["point"]
                    hip=start_points[t,feet[i]["profile"]["upper"]]
                    if np.linalg.norm(anchor-hip)>leg+np.linalg.norm(feet[i]["offset"])*1.1:
                        continue
                    weight=min(1.,(t-start+1)/4,(end-t)/4)
                    if t:
                        previous=contact_point(output_p[t-1],output_r[t-1],feet[i],surface)
                        approach=anchor-previous
                        length=np.linalg.norm(approach)
                        max_move=.045*leg
                        if length>max_move:
                            anchor=previous+approach*(max_move/length)
                    constraints.append((i,anchor,weight))
                    break
        if constraints:
            active_frames+=1
        joint_weights=np.full(len(skeleton["parents"]),.88)
        for i,_,weight in constraints:
            profile=feet[i]["profile"]
            for joint in (profile["joint"],profile["parent"],profile["upper"]):
                joint_weights[joint]=max(joint_weights[joint],weight)
        carried=Rotation.from_rotvec(
            Rotation.from_matrix(previous_correction).as_rotvec()*joint_weights[:,None]).as_matrix()
        pose_local=carried@local[t]
        root=output_p[t,0].copy()
        p,r=forward_kinematics(root[None],pose_local[None],skeleton)
        pos,rot=p[0],r[0]
        # Gradually level a planted sole before solving its ground position.
        for i,anchor,weight in constraints:
            j=feet[i]["profile"]["joint"]
            old_up=rot[j]@surface.normal
            desired_up=unit(anchors[next(k for k in anchors if k[0]==i and k[1]<=t<k[2])]["normal"])
            axis=np.cross(old_up,desired_up)
            sine=np.linalg.norm(axis)
            turn=np.arctan2(sine,np.clip(np.dot(old_up,desired_up),-1.,1.))
            if sine>1e-7:
                delta=Rotation.from_rotvec(axis/sine*turn*weight).as_matrix()
                parent=skeleton["parents"][j]
                pose_local[j]=(rot[parent].T@delta@rot[parent])@pose_local[j]
                p,r=forward_kinematics(root[None],pose_local[None],skeleton)
                pos,rot=p[0],r[0]
        for iteration in range(20 if constraints else 0):
            joints=sorted({j for i,_,_ in constraints for j in
                           (feet[i]["profile"]["joint"],feet[i]["profile"]["parent"],feet[i]["profile"]["upper"])})
            columns={j:3*n for n,j in enumerate(joints)}
            matrix=np.zeros((3*len(constraints),3*len(joints)+3))
            error=np.zeros(3*len(constraints))
            leg=min(feet[i]["profile"]["leg_length"] for i,_,_ in constraints)
            for c,(i,target,weight) in enumerate(constraints):
                foot=feet[i]
                point=contact_point(pos,rot,foot,surface)
                error[c*3:c*3+3]=(target-point)*weight
                for joint,gain in ((foot["profile"]["joint"],.25),
                                   (foot["profile"]["parent"],.7),
                                   (foot["profile"]["upper"],1.)):
                    matrix[c*3:c*3+3,columns[joint]:columns[joint]+3]=(
                        -skew(point-pos[joint])*gain)
                matrix[c*3:c*3+3,-3:]=np.eye(3)*leg*0.
            if np.max(np.abs(error))<.001*leg:
                break
            damping=(.025*leg)**2
            step=matrix.T@np.linalg.solve(matrix@matrix.T+np.eye(len(error))*damping,error)
            for joint in joints:
                gain={feet[i]["profile"]["joint"]:.25 for i,_,_ in constraints}
                omega=step[columns[joint]:columns[joint]+3]
                # The Jacobian columns were scaled to favor proximal joints.
                if joint in gain:
                    omega*=.25
                elif any(joint==feet[i]["profile"]["parent"] for i,_,_ in constraints):
                    omega*=.7
                norm=np.linalg.norm(omega)
                if norm>np.radians(6):
                    omega*=np.radians(6)/norm
                parent=skeleton["parents"][joint]
                delta=Rotation.from_rotvec(omega).as_matrix()
                pose_local[joint]=(rot[parent].T@delta@rot[parent])@pose_local[joint]
            root_step=step[-3:]*leg*.15
            size=np.linalg.norm(root_step)
            if size>leg*.025:
                root_step*=leg*.025/size
            root+=root_step
            p,r=forward_kinematics(root[None],pose_local[None],skeleton)
            pos,rot=p[0],r[0]
        if t:
            original_step=Rotation.from_matrix(
                rotations[t-1].swapaxes(-1,-2)@rotations[t]).magnitude()
            previous_angle=np.linalg.norm(
                Rotation.from_matrix(previous_correction).as_rotvec(),axis=1)
            correction_angle=np.linalg.norm(
                Rotation.from_matrix(pose_local@local[t].swapaxes(-1,-2)).as_rotvec(),axis=1)
            affected=np.zeros(len(skeleton["parents"]),dtype=bool)
            projected=np.empty_like(pose_local)
            for joint,parent in enumerate(skeleton["parents"]):
                affected[joint]=(correction_angle[joint]>1e-5 or
                                 previous_angle[joint]>1e-5 or
                                 (parent>=0 and affected[parent]))
                candidate=(projected[parent]@pose_local[joint]) if parent>=0 else pose_local[joint]
                if affected[joint]:
                    change=Rotation.from_matrix(output_r[t-1,joint].T@candidate).as_rotvec()
                    angle=np.linalg.norm(change)
                    cap=min(np.radians(45),max(np.radians(5),original_step[joint]+np.radians(3)))
                    if angle>cap:
                        candidate=output_r[t-1,joint]@Rotation.from_rotvec(change*cap/angle).as_matrix()
                        pose_local[joint]=(projected[parent].T@candidate) if parent>=0 else candidate
                projected[joint]=candidate
            p,r=forward_kinematics(root[None],pose_local[None],skeleton)
            pos,rot=p[0],r[0]
        output_p[t],output_r[t]=pos,rot
        previous_correction=pose_local@local[t].swapaxes(-1,-2)
        previous_root_shift=root-positions[t,0]
    return output_p,output_r,anchors,active_frames

def preserve_bend(positions, rotations, source_positions, skeleton, feet):
    """Keep a two-bone knee/elbow on the side chosen by the generated pose.

    Planted-foot IK can swing a limb around the hip-ankle line, and pull a
    nearly straight limb across its pole so the knee reverses as the pelvis
    turns around the fixed foot. Every frame reconstructs the knee in the
    generated pose's swivel plane, preserving both segment lengths and the
    ankle pose. Correcting only frames past a threshold made the knee pop
    between corrected and uncorrected frames.
    """
    def align(start, target):
        a, b = unit(start), unit(target)
        axis = np.cross(a, b)
        sine = np.linalg.norm(axis)
        cosine = np.clip(np.dot(a, b), -1., 1.)
        if sine < 1e-9:
            if cosine > 0:
                return np.eye(3)
            helper = np.eye(3)[np.argmin(abs(a))]
            axis = unit(np.cross(a, helper))
        else:
            axis /= sine
        return Rotation.from_rotvec(axis * np.arctan2(sine, cosine)).as_matrix()

    pos, rot = positions.copy(), rotations.copy()
    local = to_local(rot, skeleton["parents"])
    heads = np.asarray(skeleton["heads"])
    counts = {}
    for foot in feet:
        profile = foot["profile"]
        upper, lower, ankle = (profile[key] for key in ("upper", "parent", "joint"))
        length_a = np.linalg.norm(heads[lower] - heads[upper])
        length_b = np.linalg.norm(heads[ankle] - heads[lower])
        leg = length_a + length_b
        corrected = 0
        for t in range(len(pos)):
            h, k, a = pos[t, [upper, lower, ankle]]
            direction = a - h
            reach = np.linalg.norm(direction)
            if not 1e-7 < reach < leg - 1e-7:
                continue
            direction /= reach
            sh, sk, sa = source_positions[t, [upper, lower, ankle]]
            source_axis = unit(sa - sh)
            source_pole = sk - sh - source_axis * np.dot(sk - sh, source_axis)
            if np.linalg.norm(source_pole) < .002 * leg:
                continue
            pole = source_pole - direction * np.dot(source_pole, direction)
            if np.linalg.norm(pole) < 1e-7:
                continue
            pole = unit(pole)
            along = (length_a ** 2 - length_b ** 2 + reach ** 2) / (2 * reach)
            radius = np.sqrt(max(length_a ** 2 - along ** 2, 0.))
            if radius < 1e-7:
                continue
            target_k = h + direction * along + pole * radius
            if np.linalg.norm(target_k - k) < 1e-4 * leg:
                continue
            upper_new = align(k - h, target_k - h) @ rot[t, upper]
            lower_new = align(a - k, a - target_k) @ rot[t, lower]
            upper_parent = skeleton["parents"][upper]
            local[t, upper] = (rot[t, upper_parent].T @ upper_new
                               if upper_parent >= 0 else upper_new)
            local[t, lower] = upper_new.T @ lower_new
            local[t, ankle] = lower_new.T @ rot[t, ankle]
            frame_p, frame_r = forward_kinematics(pos[t:t+1, 0], local[t:t+1], skeleton)
            pos[t], rot[t] = frame_p[0], frame_r[0]
            corrected += 1
        counts[skeleton["bone_names"][ankle]] = corrected
    return pos, rot, counts


def evaluate(positions,rotations,feet,surface,windows,anchors):
    after_slides,errors=[],[]
    unresolved=0
    for (i,start,end),record in anchors.items():
        if end-start<2:
            continue
        soles=sole_points(positions[start:end],rotations[start:end],feet[i],surface)
        after_slides.extend(np.linalg.norm(np.diff(soles,axis=0),axis=1).tolist())
        distances=np.linalg.norm(soles-record["point"],axis=1)
        errors.extend(distances.tolist())
        unresolved+=int(np.sum(distances>.12*feet[i]["profile"]["leg_length"]))
    return dict(median_planted_step=float(np.median(after_slides)) if after_slides else 0.,
                max_planted_step=float(max(after_slides)) if after_slides else 0.,
                max_anchor_error=float(max(errors)) if errors else 0.,
                unresolved_contact_frames=unresolved)

def plant(positions,rotations,skeleton,ground):
    profiles=profiles_for(skeleton)
    feet=foot_capsules(skeleton,profiles)
    if not feet:
        return positions,rotations,dict(method="none",reason="No weighted foot or paw bones")
    if ground is None:
        rest=min(min(c["a"][2],c["b"][2])-c["radius"]
                 for c in skeleton["collision_capsules"] if c["joint"] in {p["joint"] for p in profiles})
        ground=dict(normal=[0.,0.,1.],height=rest,triangles=[])
    surface=Surface(ground,skeleton)
    pos,rot=positions.copy(),rotations.copy()
    # Lift deep penetrations gradually so the IK can preserve the supplied root path.
    raw=foot_metrics(pos,rot,feet,surface)
    intrusion=np.maximum.reduce([np.maximum(-item["height"],0) for item in raw])
    envelope=gaussian_filter1d(maximum_filter1d(intrusion,size=7,mode="nearest"),2,mode="nearest")
    lift=np.maximum(intrusion,envelope)
    pos+=lift[:,None,None]*surface.normal
    windows=stance_windows(pos,rot,feet,surface)
    pos,rot,limits=stabilize_feet(pos,rot,skeleton,feet,surface,windows)
    before=foot_metrics(pos,rot,feet,surface)
    before_steps=[np.linalg.norm(np.diff(item["soles"][s:e],axis=0),axis=1)
                  for item,segments in zip(before,windows) for s,e in segments if e-s>1]
    source_positions=pos.copy()
    pos,rot,anchors,active=solve_contacts(pos,rot,skeleton,feet,surface,windows)
    pos,rot,bend_corrections=preserve_bend(pos,rot,source_positions,skeleton,feet)
    metrics=evaluate(pos,rot,feet,surface,windows,anchors)
    report=dict(method="ground contact with limb IK",ground=ground.get("object","rest sole plane"),
                stance_windows={skeleton["bone_names"][foot["profile"]["joint"]]:segments
                                for foot,segments in zip(feet,windows)},
                active_frames=active,maximum_root_lift=float(max(lift)),
                median_planted_step_before=float(np.median(np.concatenate(before_steps))) if before_steps else 0.,
                missed_surface_queries=surface.misses,foot_limits=limits,
                bend_corrections=bend_corrections,**metrics)
    return pos,rot,report
