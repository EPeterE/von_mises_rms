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
from numpy import array

from pyNastran.op2.resultObjects.op2_Objects import scalarObject
from pyNastran.op2.resultObjects.tableObject import TableObject, ComplexTableObject
from pyNastran.f06.f06_formatting import writeFloats13E, writeImagFloats13E


class EigenVectorObject(TableObject):  # approach_code=2, sort_code=0, thermal=0
    """
    ::

      EIGENVALUE =  6.158494E+07
          CYCLES =  1.248985E+03         R E A L   E I G E N V E C T O R   N O .          1
  
      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3
             1      G      2.547245E-17  -6.388945E-16   2.292728E+00  -1.076928E-15   2.579163E-17   0.0
          2002      G     -6.382321E-17  -1.556607E-15   3.242408E+00  -6.530917E-16   1.747180E-17   0.0
          2003      G      0.0            0.0            0.0            0.0            0.0            0.0
    """
    def __init__(self, data_code, is_sort1, isubcase, imode):
        TableObject.__init__(self, data_code, is_sort1, isubcase, imode)
        #self.caseVal = mode
        self.update_dt = self.update_mode
        #print "mode = %s" %(mode)
        #print "data_code = ",self.data_code
        self.set_data_members()

        #assert mode>=0.
        self.gridTypes = {}
        self.translations = {imode: {}}
        self.rotations    = {imode: {}}

    def read_f06_data(self, data_code, data):
        imode = data_code['mode']
        if imode not in self.translations:
            self.update_mode(data_code, imode)

        for line in data:
            (nid, gridType, t1, t2, t3, r1, r2, r3) = line
            self.gridTypes[nid] = gridType
            self.translations[imode][nid] = array([t1, t2, t3])
            self.rotations[imode][nid] = array([r1, r2, r3])
        assert self.eigrs[-1] == data_code['eigr']

    def update_mode(self, data_code, imode):
        """
        this method is called if the object
        already exits and a new time step is found
        """
        #assert mode>=0.
        self.data_code = data_code
        self.apply_data_code()
        #self.caseVal = imode
        #print "mode = %s" %(str(mode))
        self.add_new_mode(imode)
        self.set_data_members()

    def add_new_mode(self, imode):
        self.translations[imode] = {}
        self.rotations[imode] = {}

    def eigenvalues(self):
        return self.eigrs

    def write_matlab(self, isubcase, f=None, is_mag_phase=False):
        name = 'eigenvectors'
        return self._write_matlab_transient(name, isubcase, f, is_mag_phase)

    def write_f06(self, header, pageStamp, pageNum=1, f=None, is_mag_phase=False):
        """
        ::

          EIGENVALUE =  6.158494E+07
              CYCLES =  1.248985E+03         R E A L   E I G E N V E C T O R   N O .          1
  
          POINT ID.   TYPE          T1             T2             T3             R1             R2             R3
                 1      G      2.547245E-17  -6.388945E-16   2.292728E+00  -1.076928E-15   2.579163E-17   0.0
              2002      G     -6.382321E-17  -1.556607E-15   3.242408E+00  -6.530917E-16   1.747180E-17   0.0
              2003      G      0.0            0.0            0.0            0.0            0.0            0.0
        """
        msg = []
        hasCycle = hasattr(self, 'mode_cycle')
        
        #print "self.eigrs =", self.eigrs
        #print "dir",dir(self)
        for i, (iMode, eigVals) in enumerate(sorted(self.translations.items())):
            msg += header
            freq = self.eigrs[i]
            msg.append('%16s = %13E\n' % ('EIGENVALUE', freq))

            if hasCycle:
                msg.append('%16s = %13E         R E A L   E I G E N V E C T O R   N O . %10i\n \n' % ('CYCLES', self.mode_cycle, iMode))
            else:
                msg.append('                                         R E A L   E I G E N V E C T O R   N O . %10i\n \n' % (iMode))

            msg.append('      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3\n')
            for nodeID, displacement in sorted(eigVals.items()):
                rotation = self.rotations[iMode][nodeID]
                gridType = self.gridTypes[nodeID]
                (dx, dy, dz) = displacement
                (rx, ry, rz) = rotation

                vals = [dx, dy, dz, rx, ry, rz]
                (vals2, isAllZeros) = writeFloats13E(vals)
                [dx, dy, dz, rx, ry, rz] = vals2
                msg.append('%14i %6s     %13s  %13s  %13s  %13s  %13s  %-s\n' % (nodeID, gridType, dx, dy, dz, rx, ry, rz.rstrip()))
            msg.append(pageStamp + str(pageNum) + '\n')
            if f is not None:
                f.write(''.join(msg))
                msg = ['']
            pageNum += 1
        return (''.join(msg), pageNum - 1)

    def __repr__(self):
        msg = '---EIGENVECTORS---\n'
        msg += self.print_data_members()
        name = self.data_code['name']

        headers = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
        headerLine = '%-8s %8s ' % ('nodeID', 'GridType')
        for header in headers:
            headerLine += '%10s ' % header
        headerLine += '\n'

        for i, (iMode, eigVals) in enumerate(sorted(self.translations.items())):
            freq = self.eigrs[i]
            msg += '%s = %g\n' % (name, iMode)
            msg += 'eigenvalueReal = %g\n' % freq
            msg += headerLine
            for nodeID, displacement in sorted(eigVals.items()):
                rotation = self.rotations[iMode][nodeID]
                gridType = self.gridTypes[nodeID]
                (dx, dy, dz) = displacement
                (rx, ry, rz) = rotation

                msg += '%-8i %8s ' % (nodeID, gridType)
                vals = [dx, dy, dz, rx, ry, rz]
                for val in vals:
                    if abs(val) < 1e-6:
                        msg += '%10s ' % 0
                    else:
                        msg += '%10.3g ' % val
                msg += '\n'
            msg += '\n'
            #print msg
        return msg


class RealEigenVectorObject(scalarObject):  # approach_code=2, sort_code=0, thermal=0
    """
    ::

                                           R E A L   E I G E N V E C T O R   N O .          1
        POINT ID.   TYPE          T1             T2             T3             R1             R2             R3
               1      G      0.0            0.0            0.0            0.0            1.260264E-01   0.0
               7      G      9.999849E-01   0.0            6.728968E-03   0.0            8.021386E-03   0.0
    """
    def __init__(self, data_code, isubcase, iMode):
        scalarObject.__init__(self, data_code, isubcase)
        #self.caseVal = mode
        #print "mode = %s" %(iMode)
        self.caseVal = self.getUnsteadyValue()
        self.set_data_members()

        #assert mode>=0.
        self.gridTypes = {}
        self.translations = {iMode: {}}
        self.rotations = {iMode: {}}

    def add_new_mode(self, iMode):
        self.translations[iMode] = {}
        self.rotations[iMode] = {}

    def update_dt(self, data_code, dt):
        #print " self.data_code = ",self.data_code
        self.data_code = data_code
        self.apply_data_code()
        self.set_data_members()
        self.caseVal = dt

        #print "*self.data_code = ",self.data_code
        self.translations[self.caseVal] = {}
        self.rotations[self.caseVal] = {}

    def delete_transient(self, dt):
        del self.translations[dt]
        del self.rotations[dt]

    def get_transients(self):
        k = list(self.translations.keys())
        k.sort()
        return k

    def add(self, nodeID, gridType, v1, v2, v3, v4, v5, v6):
        msg = "nodeID=%s v1=%s v2=%s v3=%s" % (nodeID, v1, v2, v3)
        msg += "           v4=%s v5=%s v6=%s" % (v4, v5, v6)
        #print msg
        assert 0 < nodeID < 1000000000, msg
        #assert nodeID not in self.translations

        if gridType == 1:
            Type = 'G'
        elif gridType == 2:
            Type = 'S'
        elif gridType == 7:
            Type = 'L'
        else:
            raise ValueError('invalid grid type...gridType=%s' % (gridType))

        self.gridTypes[nodeID] = Type
        #print 'self.caseVal = %s' %(self.caseVal),type(self.caseVal)
        #print "d = ",self.translations
        self.translations[self.caseVal][nodeID] = [v1, v2, v3]
        self.rotations[self.caseVal][nodeID] = [v4, v5, v6]

    def modes(self):
        return sorted(list(self.translations.keys()))

    def eigenvalues(self):
        return self.eigrs

    def write_matlab(self, isubcase, f=None, is_mag_phase=False):
        name = 'eigenvectors'
        return self._write_matlab_transient(name, isubcase, f, is_mag_phase)

    def write_f06(self, header, pageStamp, pageNum=1, f=None, is_mag_phase=False):
        """
        ::

          EIGENVALUE =  6.158494E+07
                                             R E A L   E I G E N V E C T O R   N O .          1
  
          POINT ID.   TYPE          T1             T2             T3             R1             R2             R3
                 1      G      2.547245E-17  -6.388945E-16   2.292728E+00  -1.076928E-15   2.579163E-17   0.0
              2002      G     -6.382321E-17  -1.556607E-15   3.242408E+00  -6.530917E-16   1.747180E-17   0.0
              2003      G      0.0            0.0            0.0            0.0            0.0            0.0
        """
        msg = []
        #print self.data_code
        for i, (iMode, eigVals) in enumerate(sorted(self.translations.items())):
            msg += header
            freq = self.eigrs[i]
            msg.append('%16s = %12E\n' % ('EIGENVALUE', freq))
            msg.append('                                         R E A L   E I G E N V E C T O R   N O . %10i\n \n' % (iMode))
            msg.append('      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3\n')
            for nodeID, translation in sorted(eigVals.items()):
                rotation = self.rotations[iMode][nodeID]
                gridType = self.gridTypes[nodeID]
                (dx, dy, dz) = translation
                (rx, ry, rz) = rotation

                vals = [dx, dy, dz, rx, ry, rz]
                (vals2, isAllZeros) = writeFloats13E(vals)
                [dx, dy, dz, rx, ry, rz] = vals2
                msg.append('%14i %6s     %13s  %13s  %13s  %13s  %13s  %-s\n' % (nodeID, gridType, dx, dy, dz, rx, ry, rz.rstrip()))
            msg.append(pageStamp + str(pageNum) + '\n')
            pageNum += 1
        return (''.join(msg), pageNum - 1)

    def __repr__(self):
        msg = '---REAL EIGENVECTORS---\n'
        msg += self.print_data_members()
        name = self.data_code['name']

        headers = ['T']
        headerLine = '%-8s %8s ' % ('nodeID', 'GridType',)
        for header in headers:
            headerLine += '%10s ' % header
        headerLine += '\n'

        for iMode, eigVals in sorted(self.translations.items()):
            msg += '%s = %s\n' % (name, iMode)
            msg += headerLine
            for nodeID, translation in sorted(eigVals.items()):
                Type = self.gridTypes[nodeID]

                rotation = self.rotations[iMode][nodeID]
                (dx, dy, dz) = translation
                (rx, ry, rz) = rotation

                vals = [dx, dy, dz, rx, ry, rz]
                msg += '%-8i %8s ' % (nodeID, Type)
                for v in vals:
                    if abs(v) < 1e-6:
                        msg += '%10s ' % 0
                    else:
                        msg += '%10.3f ' % v
                msg += '\n'
            msg += '\n'
        return msg


class ComplexEigenVectorObject(ComplexTableObject):  # approach_code=2, sort_code=0, thermal=0
    def __init__(self, data_code, is_sort1, isubcase, iMode):
        ComplexTableObject.__init__(self, data_code, is_sort1, isubcase, iMode)
        self.caseVal = iMode
        self.update_dt = self.update_mode
        self.set_data_members()

        #print "mode = %s" %(mode)

        #assert mode>=0.
        #self.gridTypes = {}
        #self.translations = {iMode: {}}
        #self.rotations    = {iMode: {}}

    def update_mode(self, data_code, iMode):
        """
        this method is called if the object
        already exits and a new time step is found
        """
        #assert mode>=0.
        self.caseVal = iMode
        self.data_code = data_code
        self.apply_data_code()
        self.set_data_members()
        #print "mode = %s" %(str(mode))
        self.add_new_mode(iMode)

    def add_new_mode(self, iMode):
        self.translations[iMode] = {}
        self.rotations[iMode] = {}

    def eigenvalues(self):
        return sorted(list(self.translations.keys()))

    def write_matlab(self, isubcase, f=None, is_mag_phase=False):
        name = 'complex_eigenvectors'
        return self._write_matlab_transient(name, isubcase, f, is_mag_phase)

    def write_f06(self, header, pageStamp, pageNum=1, f=None, is_mag_phase=False):
        msg = []
        #print self.data_code
        hasCycle = hasattr(self, 'mode_cycle')
        for i, (iMode, eigVals) in enumerate(sorted(self.translations.items())):
            msg += header
            freq = self.eigrs[i]
            #freq = 0.0
            msg.append('%16s = %12E\n' % ('EIGENVALUE', freq))
            if hasCycle:
                msg.append('%16s = %12E          C O M P L E X   E I G E N V E C T O R   N O . %10i\n \n' % ('CYCLES', self.mode_cycle, iMode))
            else:
                msg.append('                                         C O M P L E X   E I G E N V E C T O R   N O . %10i\n \n' % (iMode))
            msg.append('      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3\n')
            for nodeID, displacement in sorted(eigVals.items()):
                rotation = self.rotations[iMode][nodeID]
                gridType = self.gridTypes[nodeID]
                (dx, dy, dz) = displacement
                (rx, ry, rz) = rotation

                vals = [dx, dy, dz, rx, ry, rz]
                (vals2, isAllZeros) = writeImagFloats13E(vals, is_mag_phase)
                [dxr, dyr, dzr, rxr, ryr, rzr, dxi, dyi,
                    dzi, rxi, ryi, rzi] = vals2

                msg.append('%14i %6s     %13s  %13s  %13s  %13s  %13s  %-s\n' % (nodeID, gridType, dxr, dyr, dzr, rxr, ryr, rzr.rstrip()))
                msg.append('%14s %6s     %13s  %13s  %13s  %13s  %13s  %-s\n' % ('', '', dxi, dyi, dzi, rxi, ryi, rzi.rstrip()))

            msg.append(pageStamp + str(pageNum) + '\n')
            if f is not None:
                f.write(''.join(msg))
                msg = ['']
            pageNum += 1
        return (''.join(msg), pageNum - 1)

    def __repr__(self):
        msg = '---EIGENVECTORS---\n'
        msg += self.print_data_members()

        headers = ['T1', 'T2', 'T3', 'R1', 'R2', 'R3']
        headerLine = '%-8s %8s ' % ('nodeID', 'GridType',)
        for header in headers:
            headerLine += '%10s ' % header
        headerLine += '\n'
        name = self.data_code['name']

        for i, (iMode, eigVals) in enumerate(sorted(self.translations.items())):
            msg += '%s = %g\n' % (name, iMode)
            msg += headerLine
            for nodeID, translation in sorted(eigVals.items()):
                rotation = self.rotations[iMode][nodeID]
                Type = self.gridTypes[nodeID]
                #(dx,dy,dz) = displacement
                #(rx,ry,rz) = rotation

                msg += '%-8i %8s ' % (nodeID, Type)
                vals = translation + rotation
                for val in vals:
                    if abs(val) < 1e-6:
                        msg += '%10s ' % 0
                    else:
                        msg += '%10s ' % str(val)
                        #msg += '%10.3g ' % val
                msg += '\n'
            msg += '\n'
        return msg
