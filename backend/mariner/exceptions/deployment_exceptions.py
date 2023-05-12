"""
Deployment related exceptions
"""


class DeploymentAlreadyExists(Exception):
    """Deployment already exists exception"""


class DeploymentNotFound(Exception):
    """Deployment not found exception"""


class DeploymentNotRunning(Exception):
    """Deployment not running exception"""


class PredictionLimitReached(Exception):
    """Prediction limit reached exception"""
