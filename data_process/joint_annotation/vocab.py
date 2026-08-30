"""Vocabulary data for Truebones / Mixamo joint-name cleaning.

Kept separate from the cleaning logic so mapping edits don't require touching
code. All maps are module-level constants; import them from the cleaner.
"""

# ---------------------------------------------------------------------------
# Prefixes to strip (order matters: longest first)
# ---------------------------------------------------------------------------
REMOVE_PREFIXES = [
    "BN_Bip01_", "BN_Bip01 ",
    "Base Human ",
    "Bip001_", "Bip001 ",
    "Bip01_", "Bip01 ",
    "Preset01_", "Preset01 ",
    "Mutant:", "Sif:",
    "BN_", "NPC_", "jt_",
    "Bn_", "bn_", "b_",
]


# ---------------------------------------------------------------------------
# Mixamo-specific mappings (humanoid rig, uses CamelCase without underscores)
# ---------------------------------------------------------------------------
MIXAMO_MAP = {
    "Hips": "Hips",
    "Spine": "Spine", "Spine1": "Spine", "Spine2": "Spine",
    "Neck": "Neck",
    "Head": "Head", "HeadTop_End": "Head End",
    "Eye": "Eye",
    "Shoulder": "Shoulder",
    "Arm": "Upper Arm", "ForeArm": "Forearm", "Hand": "Hand",
    "HandThumb": "Thumb Finger", "HandIndex": "Index Finger",
    "HandMiddle": "Middle Finger", "HandRing": "Ring Finger",
    "HandPinky": "Pinky Finger",
    # Mixamo chain UpLeg -> Leg -> Foot: the mid bone "Leg" is the shin.
    "UpLeg": "Thigh", "Leg": "Shin", "Foot": "Foot",
    "ToeBase": "Toe", "Toe_End": "Toe End",
}


# ---------------------------------------------------------------------------
# Japanese vocabulary (Alligator, Pirrana, Tukan rigs)
# ---------------------------------------------------------------------------
JAPANESE_WORDS = {
    "momo": "Thigh", "sippo": "Tail", "mune": "Chest", "hiza": "Knee",
    "hara": "Abdomen", "ashi": "Foot", "hiji": "Elbow", "koshi": "Hips",
    "kosi": "Hips", "te": "Hand", "kubi": "Neck", "atama": "Head",
    "ago": "Jaw", "kata": "Shoulder", "kao": "Head", "o": "Tail",
    # Fish-specific (Pirrana)
    "munabire": "Pectoral Fin", "era": "Gill", "obire": "Caudal Fin",
    "sebire": "Dorsal Fin", "harabire": "Pelvic Fin",
    "shiribire": "Anal Fin", "shippo": "Tail",
}

# Precomputed lowercase index for fast dispatch
JAPANESE_WORDS_LOWER = {k.lower(): v for k, v in JAPANESE_WORDS.items()}


# ---------------------------------------------------------------------------
# Canonical anatomical vocabulary (applied after prefix/side stripping)
# ---------------------------------------------------------------------------
CANONICAL = {
    # Body core
    "Pelvis": "Pelvis", "pelv": "Pelvis",
    "Spine": "Spine", "Spn": "Spine", "Spline": "Spine",
    "Ribcage": "Ribcage",
    "Neck": "Neck", "Nek": "Neck",
    "Head": "Head", "Scull": "Skull", "Skull": "Skull",
    "ScullBase": "Skull Base", "SkullBase": "Skull Base",
    "HeadNub": "Head End", "Nub": "End", "Tip": "End",
    "Hips": "Hips", "Cog": "Root",
    # Limb upper
    "Clavicle": "Shoulder", "Collarbone": "Shoulder",
    "Scapula": "Scapula",
    "UpperArm": "Upper Arm", "Upperarm": "Upper Arm",
    "Humerus": "Upper Arm",
    "Forearm": "Forearm", "ForeArm": "Forearm",
    "LowerArm": "Forearm", "Lowerarm": "Forearm",
    "Radius": "Forearm",
    "Hand": "Hand", "Hnd": "Hand", "Palm": "Palm",
    "Wrist": "Wrist", "Elbow": "Elbow", "Shoulder": "Shoulder",
    # Limb lower
    "Thigh": "Thigh", "Femur": "Thigh", "UpLeg": "Thigh",
    "UpperLeg": "Thigh", "Upperleg": "Thigh",
    "Calf": "Shin", "Tibia": "Shin", "Leg": "Leg",
    "LowerLeg": "Shin", "Lowerleg": "Shin",
    "HorseLink": "Fetlock", "LargeCannon": "Cannon",
    "Metacarpus": "Metacarpus", "PhalangesManus": "Phalanges",
    "PhalanxPrima": "Pastern",
    "Foot": "Foot", "Ankle": "Ankle", "Heel": "Heel",
    "Toe": "Toe", "Toes": "Toe", "ToeBase": "Toe", "Ball": "Toe",
    "Hoof": "Hoof", "Knee": "Knee",
    "Foreleg": "Front Leg", "Hindleg": "Hind Leg",
    # Fingers
    "Finger": "Finger", "Thumb": "Thumb Finger",
    "Pinky": "Pinky Finger", "Little": "Pinky Finger",
    "Ring": "Ring Finger",
    "Middle": "Middle Finger", "Index": "Index Finger",
    # Head features
    "Jaw": "Jaw",
    "Tongue": "Tongue", "Thouge": "Tongue", "Tone": "Tongue",
    "tunge": "Tongue", "Tunge": "Tongue",
    "Ear": "Ear",
    "Eye": "Eye", "Eyeball": "Eyeball", "EyeBall": "Eyeball",
    "Eyebrow": "Eyebrow", "Eyelid": "Eyelid",
    "Eyeleds": "Eyelid",
    "Mouth": "Mouth", "Lip": "Lip",
    "Nose": "Nose", "Muzzle": "Muzzle",
    "Chin": "Chin", "Cheek": "Cheek",
    "Torso": "Body", "LegAnkle": "Ankle", "Fingers": "Finger",
    # Appendages
    "Tail": "Tail", "Tai": "Tail",
    "Wing": "Wing", "RWing": "Right Wing", "LWing": "Left Wing",
    "Feather": "Feather",
    # Species-driven, NOT typos — do not unify: Cricket "Feeler" bones are
    # antennae, Catfish "Feelers" are barbels.
    "Feeler": "Antenna", "Antenna": "Antenna", "Feelers": "Barbel",
    "Tentacle": "Tentacle", "Tentacles": "Tentacle",
    "Claw": "Claw", "HandClaw": "Hand Claw",
    "Fang": "Fang", "Fangs": "Fang",
    "Mandible": "Mandible", "BigMandible": "Large Mandible",
    "LowerMandible": "Lower Mandible",
    "Pincer": "Pincer", "pincers": "Pincer", "Pliers": "Pincer",
    "Piers": "Pincer",
    # Anatomy extras
    "Mane": "Mane", "Fur": "Fur", "Hair": "Mane",
    "Beard": "Whisker", "Mascara": "Whisker",
    "Shell": "Shell", "Stinger": "Stinger", "Horn": "Horn",
    "dorsal": "Dorsal Plate", "Fin": "Fin",
    # Species-driven, NOT typos: Tukan "ponitail" is the bird's crest,
    # humanoid "Ponytail" chains are generic appendages.
    "ponitail": "Crest", "Ponytail": "Appendage",
    "Downbody": "Lower Body", "Down": "Lower Body",
    "Body": "Body",
    # Equipment (horse)
    "Reins": "Reins", "Halter": "Halter",
    # Physics/IK
    "Jiggle": "Jiggle", "TwistBone": "Twist",
    "UpperArmTwist": "Upper Arm Twist", "ForearmTwist": "Forearm Twist",
    "ThighMuscle": "Thigh Muscle", "NeckMuscle": "Neck Muscle",
    # Insect
    "Clip": "Mandible", "Wings": "Wing", "Shall": "Mandible",
    # Misc
    "Trajectory": "Root", "locator": "Root", "locator2": "Root",
    "center": "Center",
    "MagicEffectsNode": "Bone",
    "Handle": "Handle", "IK_Chain": "IK Chain",
    # Rig-noise tokens mapped to "" are DROPPED by the cleaners; a name made
    # only of them collapses to the "Bone" placeholder ('joint16.001',
    # 'internal50.L', 'l_foreleg_jnt01').
    "Joint": "", "Jnt": "", "Internal": "", "Def": "",
    "Part": "", "Armature": "",
    "Mixamorig": "", "Character": "", "Rig": "", "Bip": "",
    "Base": "", "Human": "", "Mid": "",
    "Controller": "", "Controler": "", "Ctrl": "", "Quick": "",
    "Untitled": "", "Pasted": "",
    "Bind": "", "Skeleton": "", "Reference": "", "Res": "", "Main": "",
    # UE-style per-finger segment words: chain position, not anatomy
    # ('IndexDistal', 'thumb_proximal_l' -> the finger label alone).
    "Metacarpal": "", "Proximal": "", "Intermediate": "", "Medial": "",
    "Distal": "",
    "Digit": "Finger",
    "FrontLeg": "Front Leg", "MiddleLeg": "Middle Leg", "HindLeg": "Hind Leg",
    "Crab_pincers": "Pincer",
    "SpineR": "Right Abdomen", "SpineL": "Left Abdomen",
}


# ---------------------------------------------------------------------------
# Standalone exact-match mappings (checked early in dispatch).
# A few names (locator*, Trajectory, MagicEffectsNode) appear in CANONICAL
# too: here they catch the whole-name form before any prefix/side stripping.
# ---------------------------------------------------------------------------
STANDALONE_MAP = {
    "locator": "Root", "locator2": "Root",
    "EyesBlue_2": "Eye",
    "MagicEffectsNode": "Bone",
    "Handle": "Handle", "IK_Chain01": "IK Chain",
    "BN_P": "Belly",
    "Hips": "Hips", "Spine": "Spine", "Head": "Head",
    "Trajectory": "Root",
    "RightArm": "Right Upper Arm", "RightForeArm": "Right Forearm",
    "LeftArm": "Left Upper Arm", "LeftForeArm": "Left Forearm",
    # Standalone mixamo-style chain: bare "Leg" is the shin (see MIXAMO_MAP).
    "RightLeg": "Right Shin", "LeftLeg": "Left Shin",
    "RightUpLeg": "Right Thigh", "LeftUpLeg": "Left Thigh",
    "RightFoot": "Right Foot", "LeftFoot": "Left Foot",
    "RightHand": "Right Hand", "LeftHand": "Left Hand",
    "Tail01": "Tail", "Tail02": "Tail",
}


# ---------------------------------------------------------------------------
# SabreToothTiger: full explicit mapping (abbreviation codes are too messy)
# ---------------------------------------------------------------------------
SABRECAT_MAP = {
    "Sabrecat__pelv_": "Pelvis",
    "Sabrecat_LeftThigh_LThi_": "Left Thigh",
    "Sabrecat_LeftCalf_LClf_": "Left Shin",
    "Sabrecat_LeftFoot_LFot_": "Left Foot",
    "Sabrecat_LeftToe0_LT00_": "Left Toe",
    "Sabrecat_LeftToe0_LT01_": "Left Toe",
    "Sabrecat_RightThigh_RThi_": "Right Thigh",
    "Sabrecat_RightCalf_RClf_": "Right Shin",
    "Sabrecat_RightFoot_RFot_": "Right Foot",
    "Sabrecat_RightToe0_RT00_": "Right Toe",
    "Sabrecat_RightToe0_RT01_": "Right Toe",
    "Sabrecat_Tail0_Tal0_": "Tail",
    "Sabrecat_Tail1_Tal1_": "Tail",
    "Sabrecat_Tail2_Tal2_": "Tail",
    "Sabrecat_Spine_Spn0_": "Spine",
    "Sabrecat_Spine_Spn1_": "Spine",
    "MagicEffectsNode": "Bone",
    "Sabrecat_Spine_Spn2_": "Spine",
    "Sabrecat_Spine_Spn3_": "Spine",
    "Sabrecat_Ribcage_Spn4_": "Ribcage",
    "Sabrecat_Ribcage_Spn1_": "Ribcage",
    "Sabrecat_Neck_Nek0_": "Neck",
    "Sabrecat_Neck_Nek1_": "Neck",
    "Sabrecat_Neck_Nek2_": "Neck",
    "Sabrecat_Head__Head_": "Head",
    "Sabrecat_Head__LEye_": "Left Eye",
    "Sabrecat_Head__RChk_": "Right Cheek",
    "Sabrecat_Head_Head__LChk_": "Left Cheek",
    "Sabrecat_HeadLeftEar_LEar_": "Left Ear",
    "Sabrecat_Head_EyeLid_HELT_": "Upper Eyelid",
    "Sabrecat_Head__REye_": "Right Eye",
    "Sabrecat_Head_jaw_": "Jaw",
    "Sabrecat_HeadEyeLid__HELB_": "Lower Eyelid",
    "Sabrecat_HeadRightEar_REar_": "Right Ear",
    "Sabrecat_Head_LM01_": "Left Mouth",
    "Sabrecat_Head_RM01_": "Right Mouth",
    "Sabrecat_RightClavicle_RClv_": "Right Shoulder",
    "Sabrecat_RightUpperArm_RUar_": "Right Upper Arm",
    "Sabrecat_RightForearm_RFar_": "Right Forearm",
    "Sabrecat_RightTwistBone_RFTB_": "Right Twist",
    "Sabrecat_RightHand_RHnd_": "Right Hand",
    "Sabrecat_RightFinger3_RF30_": "Right Finger",
    "Sabrecat_RightFinger3_RF31_": "Right Finger",
    "Sabrecat_RightFinger2_RF20_": "Right Finger",
    "Sabrecat_RightFinger2_RF21_": "Right Finger",
    "Sabrecat_Finger4_RF04_": "Right Finger",
    "Sabrecat_RightFinger1_RF10_": "Right Finger",
    "Sabrecat_RightFinger1_RF11_": "Right Finger",
    "Sabrecat_RightFinger0_RF00_": "Right Finger",
    "Sabrecat_RightFinger0_RF01_": "Right Finger",
    "Sabrecat_LeftClavicle_LClv_": "Left Shoulder",
    "Sabrecat_LeftUpperArm_LUar_": "Left Upper Arm",
    "Sabrecat_LeftForearm_LFar_": "Left Forearm",
    "Sabrecat_LeftTwistBone_LFTB_": "Left Twist",
    "Sabrecat_LeftHand_LHnd_": "Left Hand",
    "Sabrecat_Finger4_LF04_": "Left Finger",
    "Sabrecat_LeftFinger1_LF10_": "Left Finger",
    "Sabrecat_LeftFinger1_LF11_": "Left Finger",
    "Sabrecat_LeftFinger2_LF20_": "Left Finger",
    "Sabrecat_LeftFinger2_LF21_": "Left Finger",
    "Sabrecat_LeftFinger3_RF30_": "Left Finger",
    "Sabrecat_LeftFinger3_RF31_": "Left Finger",
    "Sabrecat_LeftFinger0_LF00_": "Left Finger",
    "Sabrecat_LeftFinger0_LF01_": "Left Finger",
}


# ---------------------------------------------------------------------------
# Pirrana/Tukan compound Japanese names
# ---------------------------------------------------------------------------
PIRRANA_COMPOUNDS = {
    "munabireR": "Right Pectoral Fin", "munabireL": "Left Pectoral Fin",
    "eraR": "Right Gill", "eraL": "Left Gill",
    "shippoA": "Tail", "shippoB": "Tail",
    "shiribire": "Anal Fin", "shirihireB": "Anal Fin",
    "shiribireA": "Anal Fin",
    "obire": "Caudal Fin", "obireB": "Caudal Fin", "obireA": "Caudal Fin",
    "sebire": "Dorsal Fin",
    "harabireR": "Right Pelvic Fin", "harabireL": "Left Pelvic Fin",
}

# Raw names that route through the Japanese cleaner even when the lowercase
# base isn't a direct JAPANESE_WORDS hit (Pirrana/Tukan/Alligator compounds).
JAPANESE_COMPOUND_NAMES = frozenset({
    "munabireR", "munabireL", "eraR", "eraL", "shippoA", "shippoB",
    "shiribire", "shirihireB", "shiribireA", "obire", "obireB", "obireA",
    "sebire", "harabireR", "harabireL", "locator", "locator2",
    "kosi", "kao",
})
JAPANESE_COMPOUND_ANIMALS = frozenset({"Pirrana", "Tukan", "Alligator"})


# ---------------------------------------------------------------------------
# NPC_ prefix (Bear / Skyrim-style)
# ---------------------------------------------------------------------------
NPC_DIRECT = {
    "Pelvis": "Pelvis",
    "Ribcage": "Ribcage",
    "Spine1": "Spine", "Spine2": "Spine", "Spine3": "Spine", "Spine4": "Spine",
    "Neck1": "Neck", "Neck2": "Neck",
    "Head": "Head", "Jaw": "Jaw", "Nose": "Nose",
    "UpperRightLip": "Upper Right Lip", "UpperLeftLip": "Upper Left Lip",
    "UpperLip": "Upper Lip",
    "LowerLeftLip": "Lower Left Lip", "LowerFrontLip": "Lower Front Lip",
    "LowerRightLip": "Lower Right Lip",
    "Eyebrow": "Eyebrow",
    "Leg1": "Thigh", "Leg2": "Shin", "LegAnkle": "Ankle",
    "LegBall1": "Foot",
    "Toe": "Toe",
    "Arm1": "Upper Arm", "Arm2": "Forearm",
    "ArmCollarbone": "Shoulder",
    "ArmPalm": "Hand", "ArmBall1": "Foot",
    "Arm1_UpperArmTwist1": "Upper Arm Twist",
    "Arm1_UpperArmTwist2": "Upper Arm Twist",
    "Arm2_ForearmTwist1": "Forearm Twist",
    "Arm2_ForearmTwist2": "Forearm Twist",
    "Pinky01": "Pinky Finger", "Pinky02": "Pinky Finger",
    "Ring01": "Ring Finger", "Ring02": "Ring Finger",
    "Middle01": "Middle Finger", "Middle02": "Middle Finger",
    "Thumb01": "Thumb Finger", "Thumb02": "Thumb Finger",
    "Index01": "Index Finger", "Index02": "Index Finger",
}


# ---------------------------------------------------------------------------
# Elk prefix (Deer)
# ---------------------------------------------------------------------------
ELK_MAP = {
    "Femur": "Thigh", "Tibia": "Shin",
    "LargeCannon": "Cannon", "PhalanxPrima": "Pastern",
    "RearHoof": "Rear Hoof", "FrontHoof": "Front Hoof",
    "Scapula": "Scapula", "Humerus": "Upper Arm",
    "Radius": "Forearm", "Metacarpus": "Metacarpus",
    "PhalangesManus": "Phalanges",
    "Spine1": "Spine", "Spine2": "Spine", "Spine3": "Spine",
    "Ribcage": "Ribcage",
    "Neck1": "Neck", "Neck2": "Neck", "Neck3": "Neck", "Neck4": "Neck",
    "ScullBase": "Skull Base", "Scull": "Skull",
    "Ear": "Ear", "REar": "Right Ear", "LEar": "Left Ear",
    "Jaw": "Jaw", "UpperLip": "Upper Lip",
    "Pelvis": "Pelvis", "Tail1": "Tail", "Tail2": "Tail",
}


# ---------------------------------------------------------------------------
# jt_ prefix (Pteranodon, Raptor2/3, Trex, Scorpion-2)
# ---------------------------------------------------------------------------
JT_MAP = {
    "Cog": "Root", "Spine1": "Spine", "Spine2": "Spine",
    "Hips": "Hips", "Hip": "Hip",
    "Thigh": "Thigh", "Knee": "Knee", "Ankle": "Ankle", "Foot": "Foot",
    "ToeMiddle": "Middle Toe", "ToeInner": "Inner Toe", "ToeOutter": "Outer Toe",
    "ClawMiddle": "Middle Claw", "ClawInner": "Inner Claw",
    "ClawOutter": "Outer Claw", "ClawBack": "Back Claw",
    "ClawMiddle2": "Middle Claw", "ClawInner2": "Inner Claw",
    "ClawOutter2": "Outer Claw", "ClawBack2": "Back Claw",
    "Neck1": "Neck", "Neck2": "Neck", "Neck3": "Neck",
    "Head": "Head", "Jaw": "Jaw",
    "Tongue1": "Tongue", "Tongue2": "Tongue",
    "Eye": "Eye", "EyeBall": "Eyeball",
    "Shoulder": "Shoulder", "Elbow": "Elbow", "Wrist": "Wrist",
    "WristBack": "Wrist Back", "ElbowBack": "Elbow Back",
    "Clavicle": "Shoulder",
    "FingerMiddle": "Middle Finger", "FingerInner": "Inner Finger",
    "FingerOutter": "Outer Finger",
    "HandClawMiddle": "Middle Hand Claw", "HandClawInner": "Inner Hand Claw",
    "HandClawOutter": "Outer Hand Claw",
    "Tail1": "Tail", "Tail2": "Tail", "Tail3": "Tail",
    "Tail4": "Tail", "Tail5": "Tail", "Tail6": "Tail",
    "ThighMuscle": "Thigh Muscle", "NeckMuscle": "Neck Muscle",
    "Tail01": "Tail", "Tail02": "Tail", "Tail03": "Tail",
    "Tail04": "Tail", "Tail05": "Tail", "Tail06": "Tail",
    "Tail07": "Tail", "Tail08": "Tail", "Tail09": "Tail",
    "Tail01x": "Tail Twist", "Tail02x": "Tail Twist",
    "Tail03x": "Tail Twist", "Tail04x": "Tail Twist",
    "Tail05x": "Tail Twist", "Tail06x": "Tail Twist",
    "Tail07x": "Tail Twist", "Tail08x": "Tail Twist",
    "FrontLeg1": "Front Leg", "FrontLeg2": "Front Leg",
    "FrontLeg3": "Front Leg", "FrontLeg4End": "Front Leg End",
    "MiddleLeg1": "Middle Leg", "MiddleLeg2": "Middle Leg",
    "MiddleLeg3": "Middle Leg", "MiddleLeg4End": "Middle Leg End",
    "HindLeg1": "Hind Leg", "HindLeg2": "Hind Leg",
    "HindLeg3": "Hind Leg", "HindLeg4End": "Hind Leg End",
    "BigMandible": "Large Mandible", "BigMandibleMid": "Large Mandible",
    "LowerMandible": "Lower Mandible", "LowerMandibleMid": "Lower Mandible",
    "Fangs": "Fang", "FangsMid": "Fang",
}


# ---------------------------------------------------------------------------
# Spider skeleton (unique naming)
# ---------------------------------------------------------------------------
SPIDER_MAP = {
    "_body_": "Body",
    "NPC_L_MagicNode__LMag_": "Bone",
    "ArmRCollarbone": "Right Shoulder",
    "ArmR_01_": "Right Arm", "ArmR_02_": "Right Arm",
    "ArmRClaw": "Right Claw",
    "ArmLCollarbone": "Left Shoulder",
    "ArmL_01_": "Left Arm", "ArmL_02_": "Left Arm",
    "ArmLClaw": "Left Claw",
    "Tail1": "Tail", "Tail2": "Tail", "Tail3": "Tail",
    "R_Jaw_": "Right Jaw", "L_Jaw_": "Left Jaw",
}


# ---------------------------------------------------------------------------
# Canonical label format — ONE format shared by truebones / mixamo / objaverse
#
#     label := [Left |Right ] <part> [ End]
#
# where <part> is any base label the maps above emit (qualifiers like
# Front/Hind/Inner/Middle/Outer/Upper/Lower are already baked into those
# values: 'Front Leg', 'Upper Eyelid', ...) and ' End' marks a chain tip
# (Nub/Tip/*_End sources). Two sentinels sit outside the grammar:
#   - "Bone"       placeholder for rig noise with no recognizable anatomy
#   - "_00", "12"  numeric passthrough for rigs whose bones were never named
# ---------------------------------------------------------------------------

def canonical_parts():
    """Base part labels (side stripped) that the vocabulary can emit."""
    import re as _re
    parts = set()
    for mapping in (CANONICAL, MIXAMO_MAP, NPC_DIRECT, ELK_MAP, JT_MAP,
                    SPIDER_MAP, STANDALONE_MAP, SABRECAT_MAP,
                    PIRRANA_COMPOUNDS, JAPANESE_WORDS):
        for value in mapping.values():
            if value:
                parts.add(_re.sub(r'^(Left|Right)\s+', '', value))
    return parts


def is_canonical_label(label, _parts_cache=[]):
    """True iff *label* conforms to the shared canonical format."""
    import re as _re
    if not _parts_cache:
        _parts_cache.append(canonical_parts())
    if _re.match(r'^_?\d+$', label):          # numeric passthrough
        return True
    base = _re.sub(r'^(Left|Right)\s+', '', label)
    base = _re.sub(r'\s+End$', '', base)
    return base in _parts_cache[0]
