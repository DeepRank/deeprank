import pkg_resources
import os

from nose.tools import eq_, ok_

from deeprank.parse.patch import PatchParser


_patch_path = os.path.join(pkg_resources.resource_filename('deeprank.features', ''),
                         'forcefield/patch.top')


def test_parse():
    with open(_patch_path, 'rt') as f:
        result = PatchParser.parse(f)

    ok_(len(result) > 0)
    for obj in result:
        eq_(type(obj.kwargs['CHARGE']), float)
