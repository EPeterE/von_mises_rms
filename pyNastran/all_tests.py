## GNU Lesser General Public License
## 
## Program pyNastran - a python interface to NASTRAN files
## Copyright (C) 2011-2012  Steven Doyle, Al Danial
## 
## Authors and copyright holders of pyNastran
## Steven Doyle <mesheb82@gmail.com>
## Al Danial    <al.danial@gmail.com>
## 
## This file is part of pyNastran.
## 
## pyNastran is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## pyNastran is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU Lesser General Public License
## along with pyNastran.  If not, see <http://www.gnu.org/licenses/>.
## 
import unittest

#bdf
from pyNastran.bdf.test.all_tests import *

#op2
from pyNastran.op2.test.op2_unit_tests import TestOP2

#f06
from pyNastran.f06.test.f06_test import main as F06
from pyNastran.f06.test.f06_unit_tests import TestF06


#op4
from pyNastran.op4.test.op4_test import TestOP4

#gui - just tests the imports
#import pyNastran.gui.gui


if __name__ == "__main__":
    unittest.main()
    #F06()
