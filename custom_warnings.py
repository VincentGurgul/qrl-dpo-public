import warnings


def custom_formatwarning(message, category, *args, **kwargs):
    return f'{category.__name__}: {message}\n'


class ConvergenceWarning(Warning):
    def __init__(self, message):
        super().__init__(message)
        warnings.formatwarning = custom_formatwarning
    
    @staticmethod
    def warn(message):
        warnings.warn(message, ConvergenceWarning)


class PerformanceWarning(Warning):
    def __init__(self, message):
        super().__init__(message)
        warnings.formatwarning = custom_formatwarning
    
    @staticmethod
    def warn(message):
        warnings.warn(message, PerformanceWarning)
