# Changelog

## [1.2.6]
**Enhancements & Updates**
- Updated the **internal model-loading mechanism**, improving consistency and reducing overhead in template processing.
- Added support for **customizable `k` values** in the recognition process, enabling more flexible KNN-based predictions.
- Improved overall **performance and execution efficiency**, particularly in feature extraction and prediction workflows.
- Introduced an additional **model merging method**, offering more control over how multiple template dictionaries are combined.

## [1.2.5]
**Enhancements & Refactoring**
- Introduced **model merging functionality**, allowing multiple template models to be combined seamlessly.
- Added configurable **error-catching behavior**, enabling users to control whether exceptions are automatically handled.
- **Removed `timeout` attribute** and `self._timeout`, simplifying the class initialization and reducing internal state complexity.
- Improved internal consistency and robustness when handling multiple model inputs.

## [1.2.4]
**New Features & Improvements**
- Added **decorators for error catching and runtime checking**, enhancing stability during execution.
- Introduced several **custom error classes** to provide more granular exception handling, covering model operations, feature extraction, and predictions.
- Added a **`timeout` parameter** to allow users to set execution limits for long-running operations.
- Fixed **issues with overwriting previously loaded models**, ensuring that templates are preserved correctly after loading or inputting new models.

## [1.2.3]
**Bug Fixes**
- Resolved `ModuleNotFoundError` and **import-related issues**, ensuring smoother package integration.
- Fixed initialization problems that prevented proper loading of models and internal state setup.

## [1.2.2]
**Maintenance**
- Minor internal fixes and adjustments to package metadata and structure.

## [1.2.1]
**Initial Stable Release**
- First stable release with core functionality:
  - Handwriting image preprocessing
  - Feature extraction
  - KNN-based prediction
  - Model saving and loading
