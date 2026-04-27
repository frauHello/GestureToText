import sys
import cv2
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import argparse
import numpy as np
import pandas as pd
import os
BaseOptions = python.BaseOptions
RunningMode = vision.RunningMode
#POSE_DIM = 33
TOTAL_FACE_DIM = 468
HAND_DIM = 21
mp_hands = mp.tasks.vision.HandLandmarksConnections

def unique_indices(connections):
    return sorted({c.start for c in connections} |{c.end for c in connections})

face_landmarks_lips = unique_indices(FaceLandmarksConnections.FACE_LANDMARKS_LIPS)
face_landmarks_left_eye = unique_indices(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
face_landmarks_left_eyebrow = unique_indices(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW)
face_landmarks_right_eye = unique_indices(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
face_landmarks_right_eyebrow = unique_indices(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW)

REDUCED_FACE_IDX = list(dict.fromkeys(face_landmarks_left_eye + face_landmarks_left_eyebrow + face_landmarks_lips +
                          face_landmarks_right_eye + face_landmarks_right_eyebrow ))


def list_videos(root_folder: Path) -> List[Path]:
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sorted(
        [
            video_path for video_path in root_folder.rglob("*")
            if video_path.is_file() and video_path.suffix.lower() in video_exts
        ]
    )




def landmarks_to_xyz(landmarks, expected_num: int) -> np.ndarray:
        """
        Convert MediaPipe landmarks to (expected_num, 3).
        Missing landmarks are NaN.
        """
        xyz = np.full((expected_num, 3), np.nan, dtype=np.float32)

        if not landmarks:
            return xyz

        n = min(len(landmarks), expected_num)
        for i in range(n):
            lm = landmarks[i]
            xyz[i, 0] = lm.x
            xyz[i, 1] = lm.y
            xyz[i, 2] = lm.z
        return xyz

def select_reduced_face(face_landmarks) -> tuple[np.ndarray, bool]:
        """
        Extract only REDUCED_FACE_IDX from full face landmarks.
        """
        full_xyz = landmarks_to_xyz(face_landmarks, expected_num=max(REDUCED_FACE_IDX) + 1)
        
        reduced_xyz = full_xyz[REDUCED_FACE_IDX] 
        
        return reduced_xyz.astype(np.float32)

def pack_hands(hand_landmarks_list) -> tuple[np.ndarray, np.ndarray]:
        """
        Build a fixed-size 2-hand vector.

        Strategy:
        - Convert each detected hand to (21, 3)
        - Sort hands by mean x coordinate
        - Keep at most 2 hands
        - Concatenate into shape (2 * 21 * 3,)
        - Missing slots are NaN

        Returns:
            hands_vec: (126,)
            hands_valid: (2,) bool
        """
        empty_hand = np.full((HAND_DIM, 3), np.nan, dtype=np.float32)

        detected_hands: List[np.ndarray] = []
        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list[:2]:
                xyz = landmarks_to_xyz(hand_landmarks, expected_num=HAND_DIM)
                detected_hands.append(xyz)

        # Sort detected hands by horizontal position
        if len(detected_hands) > 1:
            detected_hands.sort(key=lambda h: np.nanmean(h[:, 0]))

        slots = [empty_hand.copy(), empty_hand.copy()]
        

        for i, hand_xyz in enumerate(detected_hands[:2]):
            slots[i] = hand_xyz
            
        
        hands_vec = np.concatenate(slots, axis=0).astype(np.float32)
        #hand_mask = np.isfinite(hands_vec).all(axis=-1)   # shape: (Nh,)
        return hands_vec

def extract_face_hands_keypoints(
  video_path: Path,
  face_model_path: str = "models/face_landmarker_v2_with_blendshapes.task",   
  hand_model_path: str = "models/hand_landmarker.task")-> Dict[str, np.ndarray]:
    """
    This function returns 
    - face_vectors: (num_frames, len(Reduced_face_idx)*3)
    - hands_vectors: (num_frames, 2*21*3): a fixed size vector with up to 2 hands per frame.
    """
    face_model_path = Path(face_model_path)
    hand_model_path = Path(hand_model_path)    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    REDUCED_FACE_NUM = len(REDUCED_FACE_IDX)
    face_vectors = []
    hands_vectors = []

    face_valid = []
    hands_valid = []
    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(face_model_path)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.0,
        min_tracking_confidence=0.7,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    hand_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )

    frame_count = 0

    with (
        vision.FaceLandmarker.create_from_options(face_options) as face_landmarker,
        vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker,
    ):
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb,
            )

            timestamp_ms = int((frame_count / fps) * 1000.0)
            

            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            # -------- Face --------
            if len(face_result.face_landmarks) > 0:
                face_vec = select_reduced_face(face_result.face_landmarks[0])
            else:
                face_vec = np.full((REDUCED_FACE_NUM,3), np.nan, dtype=np.float32)
                

            # ---- Hands ----
            hands_vec = pack_hands(hand_result.hand_landmarks)

            face_vectors.append(face_vec)
            hands_vectors.append(hands_vec)

           

            frame_count += 1

    cap.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames processed from video: {video_path}")

    return {
        "face_vectors": np.asarray(face_vectors, dtype=np.float32),
        "hands_vectors": np.asarray(hands_vectors, dtype=np.float32),
        "fps": np.float32(fps),
        "num_frames": np.int32(frame_count),
        "width": np.int32(width),
        "height": np.int32(height)
        }
     

def save_landmarks(output_path: Path, payload: Dict, video_path: Path):
    
    try:
        np.savez_compressed(output_path,
        face = payload["face_vectors"],   
        hands = payload["hands_vectors"]
        )
    except Exception as e:
            print(f"Failed on {str(video_path)}: {e}")
    print("saved!")



def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract normalized face, hands features from frontal RGB video clips"
    )    
    parser.add_argument("--input_dir", type=str, default="dataset/val_rgb_front_clips/raw_videos", help="Directory containing frontal video clips")
    parser.add_argument("--output_dir", type=str, default="dataset/val_rgb_front_clips/processed_features",help= "Directory where keypoint data files are stored")
    parser.add_argument("--csv_file", type=str, default="dataset/how2sign_realigned_val.csv",help= "csv file of video names")
    return parser.parse_args()
    

def main():
    args=parse_args()
   
    input_dir = Path(args.input_dir)
    output_dir=Path(args.output_dir)
    csv_path = Path(args.csv_file)
    df = pd.read_csv(csv_path , sep="\t")
    videos_df= df["SENTENCE_NAME"].unique().tolist()
    
    for video_path in videos_df:
        
        
        out_file=output_dir / f"{video_path}.npz"
        video_file=input_dir / f"{video_path}.mp4"
        
        if os.path.exists(video_file):
            print(f"Processing ... {video_path}")
            out = extract_face_hands_keypoints(video_path=video_file)
            print(out)
            save_landmarks(output_path=out_file, payload=out, video_path=video_path)
        
        
if __name__== "__main__":
    main()
