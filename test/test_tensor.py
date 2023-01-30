import numpy as np

from homopy.tensor import *


def test_orthogonal_base():
    t = Tensor()

    orthogonal_base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tensor_base = np.vstack((t.e1, t.e2, t.e3))

    assert np.allclose(orthogonal_base, tensor_base, rtol=1e-6)


class Test_Converter():

    def test_random_mandel_to_tensor_and_back(self):
        t = Tensor()

        random_mat = np.random.rand(6, 6)
        random_mat_sym = 1 / 2 * (random_mat + random_mat.T)

        random_tensor_sym = t.mandel2tensor(random_mat_sym)
        recovered_mat = t.tensor2mandel(random_tensor_sym)

        assert np.allclose(random_mat_sym, recovered_mat, rtol=1e-6)

