import numpy as np
import tritonclient.http as httpclient


def main():
    client = httpclient.InferenceServerClient(url="localhost:8000")
    assert client.is_model_ready("multimodal_classifier"), "Model not ready"

    img = np.random.randn(1, 512).astype(np.float32)
    txt = np.random.randn(1, 768).astype(np.float32)

    inputs = [
        httpclient.InferInput("img_emb", img.shape, "FP32"),
        httpclient.InferInput("txt_emb", txt.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(img)
    inputs[1].set_data_from_numpy(txt)

    outputs = [httpclient.InferRequestedOutput("logits")]
    response = client.infer("multimodal_classifier", inputs, outputs)
    logits = response.as_numpy("logits")
    print("✅ Инференс выполнен, выходная форма:", logits.shape)


if __name__ == "__main__":
    main()
