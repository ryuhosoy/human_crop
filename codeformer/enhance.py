"""CodeFormer で画像を高解像度化する最小実装"""

from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "weights" / "codeformer.onnx"

INPUT_PATH = SCRIPT_DIR.parent / "insightface" / "images" / "person_a.jpg"
OUTPUT_PATH = SCRIPT_DIR / "output" / "person_a_enhanced.jpg"
FIDELITY = 0.7


img = cv2.imread(str(INPUT_PATH))
orig_h, orig_w = img.shape[:2]

x = cv2.resize(img, (512, 512))
x = x.astype(np.float32)[:, :, ::-1] / 255.0
x = (x.transpose(2, 0, 1) - 0.5) / 0.5
x = np.expand_dims(x, 0).astype(np.float32)
w = np.array([FIDELITY], dtype=np.float64)

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
output = session.run(None, {"x": x, "w": w})[0]

y = (output[0].transpose(1, 2, 0).clip(-1, 1) + 1) * 0.5
y = (y * 255)[:, :, ::-1].clip(0, 255).astype(np.uint8)
y = cv2.resize(y, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT_PATH), y)
print(f"保存: {OUTPUT_PATH} (shape={y.shape})")
