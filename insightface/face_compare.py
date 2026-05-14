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


def detect_and_crop_face(img: np.ndarray, save_path: str = None):
    """顔を検出して最初の顔をクロップして返す"""
    faces = app.get(img)
    if not faces:
        return None, None

    # 最初の顔を使用（複数いる場合は最大の顔を選ぶ）
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    cropped = img[y1:y2, x1:x2]

    if save_path:
        cv2.imwrite(save_path, cropped)
        print(f"クロップ保存: {save_path}")

    return cropped, face


def get_embedding(img: np.ndarray) -> np.ndarray:
    """顔の特徴ベクトルを取得"""
    _, face = detect_and_crop_face(img)
    if face is None:
        return None
    return face.embedding  # 512次元ベクトル


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def is_same_person(path1: str, path2: str, threshold: float = 0.5) -> dict:
    img1 = load_image(path1)
    img2 = load_image(path2)

    # 顔クロップ保存（確認用）
    detect_and_crop_face(img1, save_path=str(SCRIPT_DIR / "face1_cropped.jpg"))
    detect_and_crop_face(img2, save_path=str(SCRIPT_DIR / "face2_cropped.jpg"))

    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)

    if emb1 is None or emb2 is None:
        return {"error": "どちらかの画像で顔が検出できませんでした"}

    score = cosine_similarity(emb1, emb2)
    same = score >= threshold

    return {
        "similarity": round(score, 4),
        "threshold": threshold,
        "same_person": same,
        "判定": "✅ 同一人物" if same else "❌ 別人"
    }


# --- 実行 ---
if __name__ == "__main__":
    result = is_same_person(
        str(IMAGES_DIR / "person_c.jpg"),
        str(IMAGES_DIR / "person_d.jpg"),
    )
    for k, v in result.items():
        print(f"{k}: {v}")