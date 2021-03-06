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


from numpy import argsort

from pyNastran.op2.resultObjects.op2_Objects import scalarObject


class OES_Object(scalarObject):
    def __init__(self, data_code, isubcase):
        scalarObject.__init__(self, data_code, isubcase)
        self.log.debug("starting OES...element_name=%s isubcase=%s" %
                       (self.element_name, self.isubcase))
        #print self.data_code

    def isCurvatureOld(self):
        if self.stress_bits[2] == 0:
            return True
        return False

    def isCurvature(self):
        if self.s_code in [0, 1, 14, 15, 16, 17, 27, 30, 31]:  # fiber distance
            return False
        elif self.s_code in [10, 11, 26, ]:  # fiber curvature
            return True
        raise NotImplementedError('add s_code=%s' % (self.s_code))

    def isFiberDistance(self):
        return not(self.isCurvature())

    def isVonMises(self):
        #print self.stress_bits
        #iMs = not(self.isMaxShear())
        #print 'isVonMises = ',iMs
        return not(self.isMaxShear())

    def isMaxShear(self):
        #print self.stress_bits
        if self.stress_bits[4] == 0:
            #print 'isMaxShear = True'
            return True
        #print 'isMaxShear = False'
        return False

    def getOrderedETypes(self, validTypes):
        """
        :param validTypes: list of valid element types
                           e.g. ['CTRIA3', 'CTRIA6', 'CQUAD4', 'CQUAD8']
        :returns TypesOut:      the ordered list of types
        :returns orderedETypes: dictionary of Type-IDs to write
        """
        orderedETypes = {}

        #validTypes = ['CTRIA3','CTRIA6','CQUAD4','CQUAD8']
        for eType in validTypes:
            orderedETypes[eType] = []
        for eid, eType in sorted(self.eType.items()):
            #print "eType = ",eType
            assert eType in validTypes, 'unsupported eType=%s' % (eType)
            orderedETypes[eType].append(eid)

        minVals = []
        for eType in validTypes:
            vals = orderedETypes[eType]
            #print "len(%s) = %s" %(eType,len(vals))
            if len(vals) == 0:
                minVals.append(-1)
            else:
                minVals.append(min(vals))

        #print "minVals = ",minVals
        argList = argsort(minVals)

        TypesOut = []
        for i in argList:
            TypesOut.append(validTypes[i])
        #print "validTypes = %s" %(validTypes)
        #print "minVals    = %s" %(minVals)
        #print "argList    = %s" %(argList)
        #print "TypesOut   = %s" %(TypesOut)
        #print "orderedETypes.keys = %s" %(orderedETypes.keys())
        return (TypesOut, orderedETypes)


class StressObject(OES_Object):
    def __init__(self, data_code, isubcase):
        OES_Object.__init__(self, data_code, isubcase)

    def update_dt(self, data_code, dt):
        self.data_code = data_code
        self.apply_data_code()
        #assert dt>=0.
        #print "data_code=",self.data_code
        self.element_name = self.data_code['element_name']
        if dt is not None:
            self.log.debug("updating stress...%s=%s element_name=%s" %
                           (self.data_code['name'], dt, self.element_name))
            self.dt = dt
            self.add_new_transient(dt)

    def isStrain(self):
        return True

    def isStress(self):
        return False


class StrainObject(OES_Object):
    def __init__(self, data_code, isubcase):
        OES_Object.__init__(self, data_code, isubcase)

    def update_dt(self, data_code, dt):
        self.data_code = data_code
        self.apply_data_code()
        #print "data_code=",self.data_code
        self.element_name = self.data_code['element_name']
        #assert dt>=0.
        if dt is not None:
            self.log.debug("updating strain...%s=%s element_name=%s" %
                           (self.data_code['name'], dt, self.element_name))
            self.dt = dt
            self.add_new_transient(dt)

    def isStress(self):
        return False

    def isStrain(self):
        return True
