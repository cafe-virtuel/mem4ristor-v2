import pytest
import sys
import os

os.environ['PYTHONPATH'] = 'src'
retcode = pytest.main(['-v', 'tests/test_kernel.py'])
sys.exit(retcode)
