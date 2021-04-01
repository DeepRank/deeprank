from nose.tools import eq_

from deeprank.models.pair import Pair


def test_order_independency():
    # test comparing:
    pair1 = Pair(1, 2)
    pair2 = Pair(2, 1)
    eq_(pair1, pair2)

    # test hashing:
    d = {pair1: 1}
    d[pair2] = 2
    eq_(d[pair1], 2)
