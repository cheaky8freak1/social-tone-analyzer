import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("models/multimodal_classifier.onnx")
img = np.random.randn(1, 512).astype(np.float32)
txt = np.random.randn(1, 768).astype(np.float32)

outputs = sess.run(["logits"], {"img_emb": img, "txt_emb": txt})
print("✅ Инференс выполнен, форма:", outputs[0].shape)
