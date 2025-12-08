import os
import numpy as np
import asyncio
from PIL import Image
from functools import partial

from .decorators import *
from .errors import *


class HandwritingRecognizer:
    def __init__(self, error_catcher: bool=True, timeout=None):
        self.timeout = timeout
        self._error_catcher = error_catcher
        self.model = {}


    @RunTimeChecker
    @ErrorCatcher
    def preprocess_image(self, path, size=28, threshold=128):
        """
        Preprocess an image into a binary array suitable for feature extraction.

        Args:
            path (str): Path to the image file.
            size (int): Size to resize the image to (size x size). Default is 28.
            threshold (int): Pixel intensity threshold for binarization. Default is 128.

        Returns:
            np.ndarray: Preprocessed binary image array.
        
        Raises:
            InvalidImageFormatError: If the image path does not exist.
            ImageProcessingError: If opening or processing the image fails.
        """
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
        """
        Extract block-based features from a preprocessed image array.

        Args:
            arr (np.ndarray): Preprocessed binary image array.
            blocks (int): Number of blocks per dimension to divide the image into. Default is 4.

        Returns:
            np.ndarray: Feature vector of normalized block sums.

        Raises:
            FeatureError: If feature extraction fails.
        """
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
        """
        Save the current model templates to a compressed .npz file.

        Args:
            filepath (str): Path to save the .npz model file.

        Raises:
            ModelSaveError: If saving the model fails.
        """
        try:
            np.savez_compressed(filepath, **self.model)
        except Exception as e:
            raise ModelSaveError(f"Failed to save model to {filepath}") from e
        print(f"Saved model to {filepath}")


    @RunTimeChecker
    @ErrorCatcher
    def load_model(self, filepath="model.npz"):
        """
        Load model templates from a compressed .npz file.

        Args:
            filepath (str): Path of the .npz model file.

        Returns:
            dict: Loaded templates keyed by 'label_filename'.
        
        Raises:
            ModelFileNotFoundError: If the .npz file does not exist.
            ModelLoadError: If loading the model fails.
        """
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
    def input_model(self, model: dict, merge: bool = False):
        """
        Input model templates from a dictionary.

        Args:
            model (dict): The model dictionary to input.
            merge (bool): Whether to merge with the existing model.

        Returns:
            dict: Updated internal model dictionary.

        Raises:
            ModelLoadError: If model is not a dictionary.
            ModalError: If merging fails.
        """

        if not isinstance(model, dict):
            raise ModelLoadError("Input model must be a dictionary")


        if not merge:
            self.model = model
            print(f"Inputted {len(self.model)} templates")
            return self.model


        if merge:
            if not isinstance(self.model, dict):
                raise ModalError("Cannot merge because current model is not a dictionary")

            try:
                for key, value in model.items():
                    self.model[key] = value
            except Exception as e:
                raise ModalError("Failed to merge model data") from e

            print(f"Inputted and merged {len(model)} templates. Total templates: {len(self.model)}")
            return self.model


        raise ModalError("Failed to input model")


    @RunTimeChecker
    @ErrorCatcher
    def merge_models(self, models: list[dict]):
        """
        Merge multiple model dictionaries into the current model.

        Args:
            models (list): List of model dictionaries to merge.

        Returns:
            dict: Updated model dictionary after merging.
        
        Raises:
            ValueError: If input is not a list of dictionaries.
        """
        if not isinstance(models, list):
            raise ValueError("models must be a list of model dictionaries")

        merged_model = {}
        for data in models:
            if not isinstance(data, dict):
                raise ValueError("Each model must be a dictionary")

            for key, value in data.items():
                merged_model[key] = value

        print(f"Merged {len(models)} models. Total templates: {len(merged_model)}")
        return merged_model


    @RunTimeChecker
    @ErrorCatcher
    def build_templates_sync(self, label, data_dir="Data", delete_after: bool=False):
        """
        Synchronously build templates for a given class label from image files.

        Args:
            label (str): Class label folder to process.
            data_dir (str): Root directory containing class folders. Default is "Data".
            delete_after (bool): Whether to delete image files after processing. Default is False.

        Raises:
            ModalError: If building a template or deleting a file fails.
            DigitRecognizeError: If preprocessing or feature extraction fails.
        """
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
        """
    Asynchronously build templates for a specific label.

    Training data directory structure must be like:
        Data/
        ├── class1/
        │   ├── picture1.png
        │   ├── picture2.png
        │   └── picture3.png
        |
        ├── class2/
        │   ├── picture1.png
        │   └── picture2.png
        └── ...

    Args:
        label (str): The class label to build templates for.
        data_dir (str): Root directory containing class folders.
        delete_after (bool): Whether to delete image files after processing.
    
    Raises:
        ModalError: If the underlying synchronous build fails.
        DigitRecognizeError: If preprocessing or feature extraction fails.
    """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, partial(self.build_templates_sync, label, data_dir, delete_after))


    @RunTimeChecker
    @ErrorCatcher
    async def build_all_templates(self, data_dir="Data", delete_after: bool=False):
        """
        Asynchronously build templates for all class labels in the data directory.

        Args:
            data_dir (str): Root directory containing class folders. Default is "Data".
            delete_after (bool): Whether to delete image files after processing. Default is False.
        
        Raises:
            ModalError: If iterating through the data directory or building templates fails.
        """
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
        """
        Predict the class label for a feature vector using K-Nearest Neighbors.

        Args:
            feat (np.ndarray): Feature vector for the image.
            k (int): Number of neighbors to consider. Default is 7.

        Returns:
            str: Predicted class label.
        
        Raises:
            NoTemplatesError: If the model is empty.
            PredictionError: If KNN prediction fails.
        """
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
        """
        Predict the class label of an image file by preprocessing and feature extraction.

        Args:
            path (str): Path to the image file.

        Returns:
            str: Predicted class label.
        
        Raises:
            FeatureError: If the feature vector is empty.
            DigitRecognizeError: If preprocessing or feature extraction fails.
            PredictionError: If KNN prediction fails.
        """
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

    