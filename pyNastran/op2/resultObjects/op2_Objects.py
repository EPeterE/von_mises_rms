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
from __future__ import print_function
#from numpy import array
from numpy import angle
from pyNastran.op2.op2Codes import Op2Codes
from pyNastran.utils import list_print


class baseScalarObject(Op2Codes):
    def __init__(self):
        pass

    def name(self):
        return self.__class__.__name__

    def write_f06(self, header, pageStamp, pageNum=1, f=None, is_mag_phase=False):
        msg = ['write_f06 is not implemented in %s\n' % (
            self.__class__.__name__)]
        return (''.join(msg), pageNum)

    def _write_f06_transient(self, header, pageStamp,
                          pageNum=1, f=None, is_mag_phase=False):
        msg = '_write_f06_transient is not implemented in %s\n' % (
            self.__class__.__name__)
        return (''.join(msg), pageNum)


class scalarObject(baseScalarObject):
    def __init__(self, data_code, isubcase):
        assert 'nonlinear_factor' in data_code, data_code
        baseScalarObject.__init__(self)
        self.isubcase = isubcase
        self.isTransient = False
        self.dt = None
        self.data_code = data_code
        self.apply_data_code()
        #self.log.debug(self.code_information())

    def isImaginary(self):
        return bool(self.sort_bits[1])

    def _write_matlab_args(self, name, isubcase, f):
        for key, value, in sorted(self.data_code.items()):
            if key is not 'log':
                if isinstance(value, str):
                    value = "'%s'" % value
                    msg = 'fem.%s(%i).%s = %s;\n' % (
                        name, isubcase, key, value)
                elif isinstance(value, list) and isinstance(value[0], str):
                    msgTemp = "','".join(value)
                    msg = "fem.%s(%i).%s = {'%s'};\n" % (
                        name, isubcase, key, msgTemp)

                elif value is None:
                    value = "'%s'" % value
                else:
                    msg = 'fem.%s(%i).%s = %s;\n' % (
                        name, isubcase, key, value)
                f.write(msg)

    def apply_data_code(self):
        self.log = self.data_code['log']
        for key, value in sorted(self.data_code.items()):
            if key is not 'log':
                self.__setattr__(key, value)
                #self.log.debug("  key=%s value=%s" %(key,value))
                #print "  key=%s value=%s" %(key,value)
        #self.log.debug("")

    def get_data_code(self):
        msg = []
        for name in self.data_code['dataNames']:
            try:
                if hasattr(self, name + 's'):
                    vals = getattr(self, name + 's')
                    name = name + 's'
                else:
                    vals = getattr(self, name)
                msg.append('  %s = %s\n' % (name, list_print(vals)))
            except AttributeError:  # weird case...
                pass
        return msg

    def getUnsteadyValue(self):
        name = self.data_code['name']
        return self.getVar(name)

    def getVar(self, name):
        return getattr(self, name)

    def set_var(self, name, value):
        return self.__setattr__(name, value)

    def start_data_member(self, varName, valueName):
        if hasattr(self, varName):
            return True
        elif hasattr(self, valueName):
            self.set_var(varName, [])
            return True
        return False

    def append_data_member(self, varName, valueName):
        """
        this appends a data member to a variable that may or may not exist
        """
        #print "append..."
        hasList = self.start_data_member(varName, valueName)
        if hasList:
            listA = self.getVar(varName)
            if listA is not None:
                #print "has %s" %(varName)
                value = self.getVar(valueName)
                try:
                    n = len(listA)
                except:
                    print("listA = ", listA)
                    raise
                listA.append(value)
                assert len(listA) == n + 1

    def set_data_members(self):
        if 'dataNames' not in self.data_code:
            msg = 'No "transient" variable was set for %s\n' % (self.table_name)
            raise NotImplementedError(msg + self.code_information())

        for name in self.data_code['dataNames']:
            #print "name = ",name
            self.append_data_member(name + 's', name)

    def update_data_code(self, data_code):
        self.data_code = data_code
        self.apply_data_code()
        self.set_data_members()

    def print_data_members(self):
        """
        Prints out the "unique" vals of the case.
        Uses a provided list of data_code['dataNames'] to set the values for
        each subcase.  Then populates a list of self.name+'s' (by using
        setattr) with the current value.  For example, if the variable name is
        'mode', we make self.modes.  Then to extract the values, we build a
        list of of the variables that were set like this and then loop over
        then to print their values.

        This way there is no dependency on one result type having ['mode'] and
        another result type having ['mode','eigr','eigi'].
        """
        keyVals = []
        for name in self.data_code['dataNames']:
            vals = getattr(self, name + 's')
            keyVals.append(vals)
            #print "%ss = %s" %(name,vals)

        msg = ''
        for name in self.data_code['dataNames']:
            msg += '%-10s ' % (name)
        msg += '\n'

        nModes = len(keyVals[0])
        for i in range(nModes):
            for vals in keyVals:
                msg += '%-10g ' % vals[i]
            msg += '\n'
        return msg + '\n'

    def recastGridType(self, gridType):
        """converts a gridType integer to a string"""
        if gridType == 1:
            Type = 'G'  # GRID
        elif gridType == 2:
            Type = 'S'  # SPOINT
        elif gridType == 7:
            Type = 'L'  # RIGID POINT (e.g. RBE3)
        elif gridType == 0:
            Type = 'H'      # SECTOR/HARMONIC/RING POINT
        else:
            raise RuntimeError('gridType=%s' % (gridType))
        return Type

    def update_dt(self, data_code, dt):
        """
        this method is called if the object
        already exits and a new time step is found
        """
        self.data_code = data_code
        self.apply_data_code()
        raise RuntimeError('update_dt not implemented in the %s class'
                           % (self.__class__.__name__))
        #assert dt>=0.
        #print "updating dt...dt=%s" %(dt)
        if dt is not None:
            self.dt = dt
            self.add_new_transient()
