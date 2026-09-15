import unittest

import numpy as np
from scipy import sparse

from domain_triage.semantic_domain_cross_view import benjamini_hochberg, centroid_cohesion


class SemanticDomainCrossViewTests(unittest.TestCase):
    def test_centroid_cohesion_uses_mean_cosine_to_normalized_centroid(self):
        matrix = sparse.csr_matrix(
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        )

        cohesion = centroid_cohesion(matrix, [0, 1])

        self.assertAlmostEqual(cohesion, 1.0 / np.sqrt(2.0), places=6)

    def test_benjamini_hochberg_preserves_order_and_is_monotone(self):
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03])

        np.testing.assert_allclose(adjusted, [0.03, 0.04, 0.04])


if __name__ == "__main__":
    unittest.main()
