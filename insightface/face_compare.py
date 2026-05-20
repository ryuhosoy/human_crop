import cv2
import numpy as np
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = SCRIPT_DIR / "images"

# --- 初期化 ---
app = FaceAnalysis(providers=['CPUExecutionProvider'])  # GPUなら 'CUDAExecutionProvider'
app.prepare(ctx_id=0, det_size=(640, 640))


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"画像が読めません: {path}")
    return img


def detect_and_crop_face(img: np.ndarray):
    """顔を検出して最大の顔をクロップして返す"""
    faces = app.get(img)
    if not faces:
        return None, None

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    cropped_img = img[y1:y2, x1:x2]

    return cropped_img, face


def save_cropped_face(cropped_img: np.ndarray, save_path: str) -> None:
    """クロップした顔画像を保存する"""
    cv2.imwrite(save_path, cropped_img)
    print(f"クロップ保存: {save_path}")


def get_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    cosine_similarity = float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
    return cosine_similarity


def verify_faces(path1: str, path2: str, threshold: float = 0.5) -> dict:
    img1 = load_image(path1)
    img2 = load_image(path2)

    cropped_face_img1, face1 = detect_and_crop_face(img1)
    cropped_face_img2, face2 = detect_and_crop_face(img2)

    if face1 is None or face2 is None:
        return {"error": "どちらかの画像で顔が検出できませんでした"}

    save_cropped_face(cropped_face_img1, str(SCRIPT_DIR / "face1_cropped.jpg"))
    save_cropped_face(cropped_face_img2, str(SCRIPT_DIR / "face2_cropped.jpg"))

    cosine_similarity = get_cosine_similarity(face1.embedding, face2.embedding)
    is_same_person = cosine_similarity >= threshold

    return {
        "similarity": round(cosine_similarity, 4),
        "threshold": threshold,
        "same_person": is_same_person,
        "判定": "✅ 同一人物" if is_same_person else "❌ 別人"
    }


# --- 実行 ---
if __name__ == "__main__":
    result = verify_faces(
        str(IMAGES_DIR / "person_a.jpg"),
        str(IMAGES_DIR / "person_a_beside_4.jpg"),
    )
    for k, v in result.items():
        print(f"{k}: {v}")
