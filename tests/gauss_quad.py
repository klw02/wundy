import numpy as np
import wundy.first as wf

"""
Tests for Gauss quadrature implementation in 1D bar element stiffness
and internal force computations.
"""

# ------------------------------------------------------------------------------
# Helper: analytical reference for linear 1D bar element
# ------------------------------------------------------------------------------

def analytical_stiffness(A, E, L):
    """Exact stiffness matrix for a 1D linear bar element."""
    return A * E / L * np.array([[1.0, -1.0],
                                 [-1.0, 1.0]])


def analytical_internal_force(A, E, L, u):
    """Exact axial internal force (constant strain)."""
    strain = (u[1] - u[0]) / L
    return A * E * strain


# ------------------------------------------------------------------------------
# Tests for Gauss quadrature in stiffness and internal force routines
# ------------------------------------------------------------------------------

def test_elem_stiff_gauss_equivalence():
    """
    Verify that the Gauss quadrature-based stiffness matrix matches
    the analytical 1D bar element stiffness for a linear displacement field.
    """
    A, E = 2.0, 210e3
    xe = np.array([[0.0], [2.0]])  # 2m element
    ke_num = wf.elem_stiff(A, E, xe)  # computed with quadrature
    ke_exact = analytical_stiffness(A, E, 2.0)
    assert np.allclose(ke_num, ke_exact, atol=1e-8), \
        "Gauss quadrature stiffness does not match analytical stiffness"


def test_elem_stiff_convergence_two_point():
    """
    Check that increasing quadrature points (if implemented) converges
    to the exact stiffness matrix.
    """
    A, E = 1.0, 100.0
    L = 1.0
    xe = np.array([[0.0], [L]])

    # single-point integration (less accurate for nonlinear fields)
    ke_1pt = wf.elem_stiff(A, E, xe, n_gauss=1)
    ke_2pt = wf.elem_stiff(A, E, xe, n_gauss=2)
    ke_exact = analytical_stiffness(A, E, L)

    # 2-point rule should be closer to exact
    err_1 = np.linalg.norm(ke_1pt - ke_exact)
    err_2 = np.linalg.norm(ke_2pt - ke_exact)
    assert err_2 <= err_1, "2-point Gauss integration should be more accurate"


def test_elem_internal_force_gauss():
    """
    Verify that the internal force vector computed using Gauss quadrature
    reproduces the exact analytical result for a constant strain state.
    """
    A, E = 1.0, 200.0
    xe = np.array([[0.0], [1.0]])
    ue = np.array([0.0, 0.01])  # uniform extension
    fint_num = wf.elem_int_force(A, E, xe, ue)
    fint_exact = analytical_internal_force(A, E, 1.0, ue)
    assert np.isclose(fint_num, fint_exact, atol=1e-8), \
        "Internal force (Gauss integration) mismatch with analytical result"


def test_elem_internal_force_sign_and_scaling():
    """
    Check that the internal force direction and scaling behave correctly
    with positive and negative displacements.
    """
    A, E, L = 1.0, 50.0, 2.0
    xe = np.array([[0.0], [L]])

    # positive extension
    ue_pos = np.array([0.0, 0.02])
    fint_pos = wf.elem_int_force(A, E, xe, ue_pos)

    # compression
    ue_neg = np.array([0.02, 0.0])
    fint_neg = wf.elem_int_force(A, E, xe, ue_neg)

    assert np.isclose(fint_pos, -fint_neg, atol=1e-8), \
        "Internal force sign not consistent for tension/compression"


def test_gauss_quadrature_points_weights():
    """
    Validate that Gauss points and weights sum correctly
    for 1- and 2-point integration rules if exposed by helper.
    """
    if hasattr(wf, "gauss_points_weights"):
        pts1, wts1 = wf.gauss_points_weights(1)
        pts2, wts2 = wf.gauss_points_weights(2)
        assert np.isclose(np.sum(wts1), 2.0, atol=1e-12)
        assert np.isclose(np.sum(wts2), 2.0, atol=1e-12)
        assert np.all(np.abs(pts2) <= 1.0), "Gauss points must be in [-1, 1]"