import jax.numpy as jnp

def create_control_canonical_matrix(coeffs):
    """
    Creates the A matrix in control canonical form given the coefficients of a polynomial,
    with coefficients [a_{n-1}, ..., a_1, a_0] placed from left to right on the bottom row,
    and 1s on the super-diagonal.

    Args:
    coeffs (list or array): The coefficients of the polynomial [a_{n-1}, ..., a_1, a_0].

    Returns:
    jnp.ndarray: The A matrix in control canonical form.
    """
    n = len(coeffs)  # Determine the order of the polynomial/system.
    # Initialize the A matrix with zeros.
    A = jnp.zeros((n, n))
    
    # Set 1s on the super-diagonal.
    A = A.at[jnp.arange(n-1), jnp.arange(1, n)].set(1)
    
    # Place the negative coefficients on the bottom row.
    A = A.at[-1].set(-jnp.array(coeffs))
    
    return A

# Example usage
coeffs = jnp.array([1, -2, 3, -4])  # Coefficients of the polynomial s^3 + 1s^2 - 2s + 3 - 4 = 0
A = create_control_canonical_matrix(jnp.flip(coeffs))
print(A)