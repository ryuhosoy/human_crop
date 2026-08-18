"""AdaFace で、すでに切り抜いた顔画像のコサイン類似度を取る"""

from pathlib import Path
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app.common import Face
from insightface.model_zoo import model_zoo

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = SCRIPT_DIR.parent / "insightface" / "images"
MODEL_PATH = SCRIPT_DIR / "weights" / "adaface_ir_101.onnx"
MODEL_URL = "https://github.com/yakhyo/adaface-onnx/releases/download/weights/adaface_ir_101.onnx"


def ensure_model() -> None:
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"AdaFace モデルをダウンロード中: {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"保存: {MODEL_PATH}")


ensure_model()

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name

# AdaFace に姿勢推定はないので、InsightFace の 3D ランドマークを使う
LMK_MODEL = Path.home() / ".insightface" / "models" / "buffalo_l" / "1k3d68.onnx"
lmk = model_zoo.get_model(str(LMK_MODEL), providers=["CPUExecutionProvider"])
lmk.prepare(ctx_id=0)


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"画像が読めません: {path}")
    return img


def to_112(img: np.ndarray) -> np.ndarray:
    """切り抜き済み顔を 112x112 にリサイズする。"""
    return cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)


def get_embedding(cropped_bgr: np.ndarray) -> np.ndarray:
    """AdaFace 公式と同じ BGR 正規化: (x - 127.5) / 127.5"""
    blob = cv2.dnn.blobFromImage(
        to_112(cropped_bgr),
        scalefactor=1.0 / 127.5,
        size=(112, 112),
        mean=(127.5, 127.5, 127.5),
        swapRB=False,
    )
    return session.run(None, {INPUT_NAME: blob})[0].flatten()


def get_pose(cropped_bgr: np.ndarray) -> dict:
    """切り抜き画像全体を顔として、pitch / yaw / roll（度）を推定する。"""
    h, w = cropped_bgr.shape[:2]
    face = Face(bbox=np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32))
    lmk.get(cropped_bgr, face)
    pitch, yaw, roll = (float(v) for v in face.pose)
    return {
        "pitch": round(pitch, 2),
        "yaw": round(yaw, 2),
        "roll": round(roll, 2),
    }


def get_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    cosine_similarity = float(
        np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    )
    return cosine_similarity


def verify_faces(path1: str, path2: str, threshold: float = 0.4) -> dict:
    img1 = load_image(path1)
    img2 = load_image(path2)

    embedding1 = get_embedding(img1)
    embedding2 = get_embedding(img2)
    pose1 = get_pose(img1)
    pose2 = get_pose(img2)

    print("face1 の向き (pitch/yaw/roll):", pose1)
    print("face2 の向き (pitch/yaw/roll):", pose2)

    cosine_similarity = get_cosine_similarity(embedding1, embedding2)
    is_same_person = cosine_similarity >= threshold

    return {
        "similarity": round(cosine_similarity, 4),
        "threshold": threshold,
        "same_person": is_same_person,
        "判定": "✅ 同一人物" if is_same_person else "❌ 別人",
        "face1_pose": pose1,
        "face2_pose": pose2,
    }


if __name__ == "__main__":
    result = verify_faces(
        str(IMAGES_DIR / "person_a.jpg"),
        str(IMAGES_DIR / "person_a_2.jpg"),
    )
    for k, v in result.items():
        print(f"{k}: {v}")
