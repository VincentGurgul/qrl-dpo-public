'''Module containing input transformations to be applied to the data before
being fed to the predictor.

Methods:
    min_max_scale_to_range: Scales the data using min-max normalization to a
        specified range.
    normalized_arcsin: Transforms the data using min-max normalization to [-1, 1]
        and applies the arcsin function.
    standardize: Standardizes the data using z-score normalization.
    radial_to_linear: Transforms data using direct Cartesian feature augmentations
        to escape radial patterns.
'''
import torch


def min_max_scale_to_range(
    X: torch.Tensor,
    range: tuple = (-1, 1),
) -> torch.Tensor:
    '''
    Scales the data using min-max normalization to a specified range.

    Args:
        X (torch.Tensor): Input tensor to be scaled.
        range (tuple, optional): Desired range for scaling, specified as
            (min, max). Defaults to (-1, 1).

    Returns:
        torch.Tensor: Scaled tensor with values in the specified range.
            Same dimensions as input.
    '''
    # Normalize to [0, 1]
    X_min, X_max = torch.min(X, dim=0).values, torch.max(X, dim=0).values
    X_normalized = (X - X_min) / (X_max - X_min)
    
    # Scale to [min_val, max_val]
    min_val, max_val = range
    X_scaled = X_normalized * (max_val - min_val) + min_val

    return X_scaled


def normalized_arcsin(X: torch.Tensor) -> torch.Tensor:
    '''
    Transforms the data using min-max normalization to [-1, 1] and applies the arcsin function.

    Args:
        X (torch.Tensor): Input tensor to be transformed.

    Returns:
        torch.Tensor: Transformed tensor with values normalized to [-1, 1]
            and then passed through the arcsin function. Same dimensions as input.
    '''
    # Normalize to [-1, 1]
    X = min_max_scale_to_range(X, range=(-1, 1))

    # Apply arcsin
    X = torch.arcsin(X)

    return X


def standardize(X: torch.Tensor, dof: int = 0) -> torch.Tensor:
    '''
    Standardizes the data using z-score normalization.

    Args:
        X (torch.Tensor): Input tensor to be standardized.
        dof (int, optional): Degrees of freedom on which the calculation of
            the standard deviation is to be based on. Value of 1 is unbiased
            estimator, value of 0 is biased estimator. Defaults to 0.

    Returns:
        torch.Tensor: Standardized tensor with a mean of 0 and a standard
            deviation of 1 for each feature. Same dimensions as input.
    '''
    mean = torch.mean(X, dim=0)
    std = torch.std(X, dim=0, correction=dof)
    return (X - mean) / std


def radial_to_linear(
    X: torch.Tensor,
) -> torch.Tensor:
    '''
    Transforms data using direct Cartesian feature augmentations to escape
    radial patterns. Outputs a tensor of original data, original data squared,
    sine, and cosine. Hence, the output dimension is 4 times the input dimension.

    Args:
        X (torch.Tensor): Input tensor of shape (n_samples, n_features).

    Returns:
        torch.Tensor: Transformed tensor of shape (n_samples, 4 * n_features).
    '''
    # Standardize features
    X = standardize(X)

    # Augment features
    transformed_data = torch.cat([
        X,              # Original features
        X**2,           # Quadratic features
        torch.sin(X),   # Sinusoidal features
        torch.cos(X)    # Cosine features
    ], dim=-1)  # Concatenate along the feature dimension

    return transformed_data


def radial_to_linear_small(
    X: torch.Tensor,
) -> torch.Tensor:
    '''
    Transforms data using direct Cartesian feature augmentations to escape
    radial patterns. Outputs a tensor of original data and cosine (after
    standardization). Hence, the output dimension is twice the input dimension.

    Args:
        X (torch.Tensor): Input tensor of shape (n_samples, n_features).

    Returns:
        torch.Tensor: Transformed tensor of shape (n_samples, 2 * n_features).
    '''
    # Standardize features
    X = standardize(X)

    # Augment features
    transformed_data = torch.cat([
        X,                    # Original features
        torch.cos(X),         # Cosine features
    ], dim=-1)  # Concatenate along the feature dimension

    return transformed_data