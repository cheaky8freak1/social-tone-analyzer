import torch

from social_tone.models.classifier import MultimodalClassifier


def export_onnx(checkpoint_path: str, output_path: str = "models/multimodal_classifier.onnx"):
    """
    Экспорт обученной модели в ONNX.
    """
    model = MultimodalClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()

    dummy_img = torch.randn(1, 512)  # img_dim = 512
    dummy_txt = torch.randn(1, 768)  # txt_dim = 768

    torch.onnx.export(
        model,
        (dummy_img, dummy_txt),
        output_path,
        input_names=["img_emb", "txt_emb"],
        output_names=["logits"],
        dynamic_axes={
            "img_emb": {0: "batch_size"},
            "txt_emb": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"✅ ONNX модель сохранена: {output_path}")


if __name__ == "__main__":
    checkpoint_path = "checkpoints/best-epoch=06-val_loss=0.1492.ckpt"
    export_onnx(checkpoint_path)
