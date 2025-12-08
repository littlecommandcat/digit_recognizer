class DigitRecognizeError(Exception):
    """Base exception for all digit recognition errors.

    Example:
        >>> raise DigitRecognizeError("Module encountered an unexpected issue")
    """
    pass


class ModalError(DigitRecognizeError):
    """Base exception for model-related issues.

    Example:
        >>> raise ModalError("Model encountered an unexpected issue")
    """
    pass


class ModelFileNotFoundError(ModalError):
    """Raised when a model file cannot be found.

    Example:
        >>> raise ModelFileNotFoundError("Model file 'weights.dat' not found")
    """
    pass


class ModelSaveError(ModalError):
    """Raised when saving the model fails.

    Example:
        >>> raise ModelSaveError("Failed to save model to disk")
    """
    pass


class ModelLoadError(ModalError):
    """Raised when loading the model fails.

    Example:
        >>> raise ModelLoadError("Failed to load model from file")
    """
    pass


class FeatureError(DigitRecognizeError):
    """Base exception for feature extraction issues.

    Example:
        >>> raise FeatureError("Feature extraction failed")
    """
    pass


class ImageProcessingError(FeatureError):
    """Raised when image processing fails.

    Example:
        >>> raise ImageProcessingError("Image preprocessing pipeline crashed")
    """
    pass


class InvalidImageFormatError(FeatureError):
    """Raised when the input image has an invalid format.

    Example:
        >>> raise InvalidImageFormatError("Input image format not supported")
    """
    pass


class PredictionError(DigitRecognizeError):
    """Base exception for prediction-related issues.

    Example:
        >>> raise PredictionError("Prediction process encountered an error")
    """
    pass


class NoTemplatesError(PredictionError):
    """Raised when prediction templates or reference data are missing.

    Example:
        >>> raise NoTemplatesError("No templates available for prediction")
    """
    pass
