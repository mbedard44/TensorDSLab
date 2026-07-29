import unittest

import torch

from tensor_dslab import Photoelectrons
from tests._product_support import source


class PhotoelectronTests(unittest.TestCase):
    def test_source_domain(self) -> None:
        product = source()
        Photoelectrons.validate(product=product)
        with self.assertRaises(ValueError):
            Photoelectrons(
                tensor=torch.full(product.shape, -1, dtype=torch.int64),
                spec=product.spec,
            )
