import os
import pkg_resources

from nose.tools import eq_, ok_

from deeprank.parse.param import ParamParser



_param_path = os.path.join(pkg_resources.resource_filename('deeprank.features', ''),
                           "forcefield/protein-allhdg5-4_new.param")

def test_parse():
    with open(_param_path, 'rt') as f:
        result = ParamParser.parse(f)

    ok_(len(result) > 0)
    eq_(type(list(result.values())[0].epsilon), float)
