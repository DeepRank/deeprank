import pkg_resources
import os

from nose.tools import eq_, ok_

from deeprank.parse.top import TopParser


_top_path = os.path.join(pkg_resources.resource_filename('deeprank.features', ''),
                         'forcefield/protein-allhdg5-4_new.top')


def test_parse():
    with open(_top_path, 'rt') as f:
        result = TopParser.parse(f)

    eq_(len(result), 705)

    for obj in result:
        eq_(type(obj.kwargs['type']), str)
        eq_(type(obj.kwargs['charge']), float)
