from nose.tools import eq_, ok_

from deeprank.models.pair import Pair


def test_order_independency():
    # These should be the same:
    pair1 = Pair(1, 2)
    pair2 = Pair(2, 1)

    # test comparing:
    eq_(pair1, pair2)

    # test hashing:
    d = {pair1: 1}
    d[pair2] = 2
    eq_(d[pair1], 2)


def test_uniqueness():
    # These should be different:
    pair1 = Pair(1, 2)
    pair2 = Pair(1, 3)

    # test comparing:
    ok_(pair1 != pair2)

    # test hashing:
    d = {pair1: 1}
    d[pair2] = 2
    eq_(d[pair1], 1)
