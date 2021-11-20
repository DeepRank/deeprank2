from nose.tools import eq_, ok_

from deeprank_gnn.models.pair import Pair, PairTable, ContactPair


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


def test_table():
    # These should be different:
    table = PairTable()
    table[1, 2] = 0
    table[1, 3] = 1

    # test lookup:
    eq_(table[1, 2], 0)
    eq_(table[2, 1], 0)
    eq_(table[1, 3], 1)


def test_contact():
    # should be different distances
    pair1 = ContactPair(0, 1, 0.1)
    pair2 = ContactPair(1, 0, 0.1)
    pair3 = ContactPair(0, 1, 0.2)

    ok_(pair1 == pair2)
    ok_(pair1 != pair3)
