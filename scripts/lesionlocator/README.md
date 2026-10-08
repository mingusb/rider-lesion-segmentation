# External CNN adapter (GPL-3.0-or-later)

This directory wraps the separately installed GPLv3 LesionLocator at commit ef073baaa0dcba8dd036ec5a29f130db18c9d580. License text is in ../../licenses/GPL-3.0.txt. No CNN weights are included. Released checkpoints have a separate CC-BY-NC-SA 4.0 license.

The reference and extraction methods are the ones used in the completed study. Packaging changes add explicit workspace/model paths, resolve a relative exported image path, build an image-only prepared index and remove private report-file updates. The numerical feature extraction is unchanged. Source hash records and the declared protocol distinguish this packaging from the original measured environment.
