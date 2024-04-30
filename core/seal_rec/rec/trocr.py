import cv2
import numpy as np
import onnxruntime
import os
import statistics
from utils.compute.logsumexp import softmax
from core.seal_rec.rec.config import trocr_args
from core.seal_rec.rec.tool import read_vocab, decode_text
from utils.image_utils import normalize


class OnnxEncoder(object):
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())

    def __call__(self, image):
        onnx_inputs = {self.model.get_inputs()[0].name: np.asarray(image, dtype='float32')}
        onnx_output = self.model.run(None, onnx_inputs)[0]
        return onnx_output


class OnnxDecoder(object):
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path)
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.model.get_inputs())}

    def __call__(self, input_ids,
                 encoder_hidden_states,
                 attention_mask):
        input_info = {"input_ids": input_ids,
                      "attention_mask": attention_mask,
                      "encoder_hidden_states": encoder_hidden_states}
        # 兼容不同版本的模型输入 todo 未来统一模型输入值
        onnx_inputs = {key: input_info[key] for key in self.input_names}
        onnx_output = self.model.run(['logits'], onnx_inputs)
        return onnx_output


class OnnxEncoderDecoder(object):
    def __init__(self, module_args=trocr_args):
        self.encoder = OnnxEncoder(os.path.join(module_args.model_path, "encoder_model.onnx"))
        self.decoder = OnnxDecoder(os.path.join(module_args.model_path, "decoder_model.onnx"))
        self.vocab = read_vocab(os.path.join(module_args.model_path, "vocab.json"))
        self.vocab_inp = {self.vocab[key]: key for key in self.vocab}
        self.threshold = module_args.threshold
        self.max_len = module_args.max_len
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        self.input_shape = module_args.input_shape

    def rec_norm_image(self, norm_image):
        """
        norm_image:归一化后的图像数据
        """
        encoder_output = self.encoder(norm_image)
        ids = [self.vocab["<s>"], ]
        mask = [1, ]
        scores = []
        for i in range(self.max_len):
            input_ids = np.array([ids]).astype('int64')
            attention_mask = np.array([mask]).astype('int64')
            decoder_output = self.decoder(input_ids=input_ids,
                                          encoder_hidden_states=encoder_output,
                                          attention_mask=attention_mask)
            pred = decoder_output[0][0]
            pred = softmax(pred, axis=1)
            max_index = pred.argmax(axis=1)
            if max_index[-1] == self.vocab["</s>"]:
                break
            scores.append(pred[max_index.shape[0] - 1, max_index[-1]])
            ids.append(max_index[-1])
            mask.append(1)
        if self.threshold < statistics.mean(scores):
            text = decode_text(ids, self.vocab, self.vocab_inp)
        else:
            text = ""
        return text

    def rec(self, image: np.ndarray):
        """
        image: 三通道rgb图像
        """
        image = cv2.resize(image, self.input_shape)
        pixel_values = normalize(image, self.mean, self.std)
        pixel_values = np.expand_dims(np.transpose(pixel_values, (2, 0, 1)), axis=0)
        rec_text = self.rec_norm_image(pixel_values)
        return rec_text


if __name__ == '__main__':
    from utils.image_utils import read_image_file

    test_image_path = r"D:\GitHubWorkSpace\ImageAnalysisService\core\seal_rec\rec\tool.py"
    model = OnnxEncoderDecoder()
    img = read_image_file(test_image_path)
    res = model.rec(img)
    print(res)
