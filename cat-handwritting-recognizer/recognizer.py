import os
import numpy as np
import asyncio
from PIL import Image
from functools import partial

from .decoration import *
from .errors import *


class HandwritingRecognizer:
    def __init__(self, timeout=None):
        self.timeout = timeout
        self._starttime = None
        self.model = {}


    @RunTimeChecker
    @ErrorCatcher
    def preprocess_image(self, path, size=28, threshold=128):
        try:
            if not os.path.exists(path):
                raise InvalidImageFormatError(f"Image path not found {path}")

            img = Image.open(path).convert('L')
        except Exception as e:
            raise ImageProcessingError(f"Failed to open image {path}") from e

        try:
            arr = np.array(img, dtype=np.uint8)
            arr = np.where(arr > threshold, 0, 1).astype(np.uint8)

            ys, xs = np.where(arr == 1)
            if len(xs) > 0 and len(ys) > 0:
                arr = arr[min(ys):max(ys)+1, min(xs):max(xs)+1]

            img = Image.fromarray(arr * 255).resize((size, size), Image.NEAREST)
            arr = np.array(img, dtype=np.uint8)

            return np.where(arr > 128, 0, 1).astype(np.uint8)
        except Exception as e:
            raise ImageProcessingError("Image preprocessing failed") from e


    @RunTimeChecker
    @ErrorCatcher
    def extract_features(self, arr, blocks=4):
        try:
            h, w = arr.shape
            block_h = h // blocks
            block_w = w // blocks

            feats = []
            for i in range(blocks):
                for j in range(blocks):
                    block = arr[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                    feats.append(block.sum() / (block_h * block_w))

            return np.array(feats)

        except Exception as e:
            raise FeatureError("Failed to extract features") from e


    @RunTimeChecker
    @ErrorCatcher
    def save_model(self, filepath="model.npz"):
        try:
            np.savez_compressed(filepath, **self.model)
        except Exception as e:
            raise ModelSaveError(f"Failed to save model to {filepath}") from e
        print(f"Saved model to {filepath}")


    @RunTimeChecker
    @ErrorCatcher
    def load_model(self, filepath="model.npz"):
        if not os.path.exists(filepath):
            raise ModelFileNotFoundError(f"Model file not found {filepath}")

        try:
            data = np.load(filepath)
            self.model = {key: data[key] for key in data.files}
        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {filepath}") from e

        print(f"Loaded {len(self.model)} templates")
        return self.model


    @RunTimeChecker
    @ErrorCatcher
    def build_templates_sync(self, label, data_dir="Data", delete_after: bool=False):
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            return

        for fname in os.listdir(folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(folder, fname)
                try:
                    arr = self.preprocess_image(path)
                    feat = self.extract_features(arr)
                    key = f"{label}_{fname.split('.')[0]}"
                    self.model[key] = feat
                except DigitRecognizeError:
                    raise
                except Exception as e:
                    raise ModalError(f"Failed to build template for file {path}") from e

                print(f"Build template {key}")

                if delete_after:
                    try:
                        os.remove(path)
                    except Exception as e:
                        raise ModalError(f"Failed to delete file {path}") from e


    @RunTimeChecker
    @ErrorCatcher
    async def build_templates(self, label, data_dir="Data", delete_after: bool=False):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, partial(self.build_templates_sync, label, data_dir, delete_after))


    @RunTimeChecker
    @ErrorCatcher
    async def build_all_templates(self, data_dir="Data", delete_after: bool=False):
        tasks = []
        try:
            for label in os.listdir(data_dir):
                tasks.append(self.build_templates(label, data_dir, delete_after))
        except Exception as e:
            raise ModalError("Failed to iterate through data directory") from e

        await asyncio.gather(*tasks)


    @RunTimeChecker
    @ErrorCatcher
    def knn_predict(self, feat, k=7):
        if not self.model:
            raise NoTemplatesError("Model is empty. Build or load templates first.")

        try:
            distances = []
            for label, tfeat in self.model.items():
                d = np.linalg.norm(feat - tfeat)
                distances.append((d, label))

            distances.sort()
            votes = [lbl.split("_")[0] for _, lbl in distances[:k]]
            return max(set(votes), key=votes.count)

        except Exception as e:
            raise PredictionError("Prediction failed") from e


    @RunTimeChecker
    @ErrorCatcher
    async def predict(self, path):
        if not self.model:
            self.model = self.load_model()

        try:
            arr = self.preprocess_image(path)
            feat = self.extract_features(arr)
        except DigitRecognizeError:
            raise
        except Exception as e:
            raise PredictionError("Failed to preprocess image for prediction") from e

        if feat.size == 0:
            raise FeatureError("Feature vector is empty")

        return self.knn_predict(feat)
