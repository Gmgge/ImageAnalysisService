import cv2
import numpy as np
import onnxruntime
import os
import torch
import statistics
from utils.compute.logsumexp import softmax
from core.seal_rec.rec.config import trocr_args
from core.seal_rec.rec.tool import read_vocab, decode_text
from utils.image_utils import normalize
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image


class TorchEncoderDecoder(object):
    def __init__(self, module_args=trocr_args):
        self.threshold = module_args.threshold
        self.max_len = module_args.max_len
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        self.input_shape = module_args.input_shape

        self.processor = TrOCRProcessor.from_pretrained(module_args.model_path)
        self.vocab = self.processor.tokenizer.get_vocab()
        self.vocab_inp = {self.vocab[key]: key for key in self.vocab}
        self.model = VisionEncoderDecoderModel.from_pretrained(module_args.model_path)
        self.model.eval()

    def rec(self, image: np.ndarray):
        """
        image: 三通道rgb图像
        """
        pixel_values = self.processor([image], return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values[:, :, :].cpu())

        generated_text = decode_text(generated_ids[0].cpu().numpy(), self.vocab, self.vocab_inp)
        return generated_text


if __name__ == '__main__':
    from utils.image_utils import read_image_file
    test_image = r"F:\project\ImageAnalysisService\tests\core\seal\image\seal_3.jpg"
    img = read_image_file(test_image)
    seal_rec_opt = TorchEncoderDecoder()
    print(seal_rec_opt.rec(img))

