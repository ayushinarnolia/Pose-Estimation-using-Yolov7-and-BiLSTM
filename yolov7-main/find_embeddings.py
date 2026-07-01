#find_embeddings.py
import numpy as np

def keypoints_to_embedding(kpts):
    """
    Embedding:
    - 34 normalized coordinates (x,y for 17 keypoints)
    - 8 joint angles normalized to [0,1]
    Returns a 42-dimensional embedding.
    """

    kpts = np.array(kpts, dtype=np.float32)

    # Compute torso center and torso length
    shoulder_mid = (kpts[5] + kpts[6]) / 2
    hip_mid = (kpts[11] + kpts[12]) / 2
    torso_center = (shoulder_mid + hip_mid) / 2

    torso_length = np.linalg.norm(shoulder_mid - hip_mid) + 1e-8

    # Torso normalization
    kpts_norm = (kpts - torso_center) / torso_length

    # Flatten coordinates
    kpts_flat = kpts_norm.flatten()

    # Map approximate range [-2, 2] -> [0, 1]
    kpts_flat = np.clip((kpts_flat + 2.0) / 4.0, 0.0, 1.0)

    # ----------------------------
    # Helper functions
    # ----------------------------
    def angle_3pts(a, b, c):
        """Returns angle ABC in degrees, b is the vertex."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine = np.dot(ba, bc) / (
            np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
        )
        cosine = np.clip(cosine, -1.0, 1.0)

        return np.degrees(np.arccos(cosine))

    def angle_between_vectors(v1, v2):
        """Returns angle between two vectors in degrees."""
        v1 = np.array(v1)
        v2 = np.array(v2)

        cosine = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        )
        cosine = np.clip(cosine, -1.0, 1.0)

        return np.degrees(np.arccos(cosine))

    # ----------------------------
    # Keypoint indices
    # ----------------------------
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16

    # ----------------------------
    # Joint angles
    # ----------------------------

    # Knee flexion
    left_knee_flexion = 180 - angle_3pts(
        kpts[L_HIP], kpts[L_KNEE], kpts[L_ANKLE]
    )
    right_knee_flexion = 180 - angle_3pts(
        kpts[R_HIP], kpts[R_KNEE], kpts[R_ANKLE]
    )

    # Hip flexion
    left_hip_flexion = 180 - angle_3pts(
        kpts[L_SHOULDER], kpts[L_HIP], kpts[L_KNEE]
    )
    right_hip_flexion = 180 - angle_3pts(
        kpts[R_SHOULDER], kpts[R_HIP], kpts[R_KNEE]
    )

    # Elbow flexion
    left_elbow_flexion = 180 - angle_3pts(
        kpts[L_SHOULDER], kpts[L_ELBOW], kpts[L_WRIST]
    )
    right_elbow_flexion = 180 - angle_3pts(
        kpts[R_SHOULDER], kpts[R_ELBOW], kpts[R_WRIST]
    )

    # Trunk lean
    shoulder_center = (kpts[L_SHOULDER] + kpts[R_SHOULDER]) / 2
    hip_center = (kpts[L_HIP] + kpts[R_HIP]) / 2

    trunk_vec = shoulder_center - hip_center
    vertical_vec = np.array([0, -1])

    trunk_lean = angle_between_vectors(trunk_vec, vertical_vec)

    # Inter-thigh angle
    left_thigh_vec = kpts[L_KNEE] - kpts[L_HIP]
    right_thigh_vec = kpts[R_KNEE] - kpts[R_HIP]

    inter_thigh_angle = angle_between_vectors(
        left_thigh_vec,
        right_thigh_vec
    )

    # Normalize angles to [0,1]
    angles = np.array([
        left_knee_flexion,
        right_knee_flexion,
        left_hip_flexion,
        right_hip_flexion,
        left_elbow_flexion,
        right_elbow_flexion,
        trunk_lean,
        inter_thigh_angle
    ], dtype=np.float32)

    angles = np.clip(angles / 180.0, 0.0, 1.0)

    # Final embedding (34 coords + 8 angles = 42)
    embedding = np.concatenate([kpts_flat, angles])

    return embedding.astype(np.float32).tolist()

"""
0: Keypoint 0 (Nose) (x)
1: Keypoint 0 (Nose) (y)
2: Keypoint 1 (Left Eye) (x)
3: Keypoint 1 (Left Eye) (y)
4: Keypoint 2 (Right Eye) (x)
5: Keypoint 2 (Right Eye) (y)
6: Keypoint 3 (Left Ear) (x)
7: Keypoint 3 (Left Ear) (y)
8: Keypoint 4 (Right Ear) (x)
9: Keypoint 4 (Right Ear) (y)
10: Keypoint 5 (Left Shoulder) (x)
11: Keypoint 5 (Left Shoulder) (y)
12: Keypoint 6 (Right Shoulder) (x)
13: Keypoint 6 (Right Shoulder) (y)
14: Keypoint 7 (Left Elbow) (x)
15: Keypoint 7 (Left Elbow) (y)
16: Keypoint 8 (Right Elbow) (x)
17: Keypoint 8 (Right Elbow) (y)
18: Keypoint 9 (Left Wrist) (x)
19: Keypoint 9 (Left Wrist) (y)
20: Keypoint 10 (Right Wrist) (x)
21: Keypoint 10 (Right Wrist) (y)
22: Keypoint 11 (Left Hip) (x)
23: Keypoint 11 (Left Hip) (y)
24: Keypoint 12 (Right Hip) (x)
25: Keypoint 12 (Right Hip) (y)
26: Keypoint 13 (Left Knee) (x)
27: Keypoint 13 (Left Knee) (y)
28: Keypoint 14 (Right Knee) (x)
29: Keypoint 14 (Right Knee) (y)
30: Keypoint 15 (Left Ankle) (x)
31: Keypoint 15 (Left Ankle) (y)
32: Keypoint 16 (Right Ankle) (x)
33: Keypoint 16 (Right Ankle) (y)

34: Left Knee Flexion
35: Right Knee Flexion
36: Left Hip Flexion
37: Right Hip Flexion
38: Left Elbow Flexion
39: Right Elbow Flexion
40: Trunk Lean
41: Inter-Thigh Angle"""