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



from pyNastran.utils import is_string
from .oes_objects import StressObject, StrainObject
from pyNastran.f06.f06_formatting import writeFloats13E, writeImagFloats13E


class BarStressObject(StressObject):
    """
    ::
    
      # s_code=0
                                 S T R E S S E S   I N   B A R   E L E M E N T S          ( C B A R )
      ELEMENT        SA1            SA2            SA3            SA4           AXIAL          SA-MAX         SA-MIN     M.S.-T
        ID.          SB1            SB2            SB3            SB4           STRESS         SB-MAX         SB-MIN     M.S.-C
    """
    def __init__(self, data_code, is_sort1, isubcase, dt=None):
        StressObject.__init__(self, data_code, isubcase)
        self.eType = {}

        self.code = [self.format_code, self.sort_code, self.s_code]

        self.s1 = {}
        self.s2 = {}
        self.s3 = {}
        self.s4 = {}
        self.axial = {}
        self.smax = {}
        self.smin = {}
        self.MS_tension = {}
        self.MS_compression = {}

        #if self.element_type==100:
            #self.getLength = self.getLength100_format1_sort0
            #self.add_new_eid = self.addNewEid100

        self.dt = dt
        #print "BAR dt=%s" %(dt)
        if is_sort1:
            if dt is not None:
                #self.add = self.add_sort1
                self.add_new_eid = self.add_new_eid_sort1
        else:
            assert dt is not None
            #self.add = self.addSort2
            self.add_new_eid = self.add_new_eid_sort2

    def get_stats(self):
        nelements = len(self.eType)

        msg = self.get_data_code()
        if self.dt is not None:  # transient
            ntimes = len(self.s1)
            msg.append('  type=%s ntimes=%s nelements=%s\n'
                       % (self.__class__.__name__, ntimes, nelements))
        else:
            msg.append('  type=%s nelements=%s\n' % (self.__class__.__name__,
                                                     nelements))
        msg.append('  eType, s1, s2, s3, s4, axial, smax, smin, '
                   'MS_tension, MS_compression\n')
        return msg

    def add_f06_data(self, data, transient):
        if transient is None:
            for line in data:
                (eType, eid, s1A, s2A, s3A, s4A, axialA, smaxA, sminA, MSt,
                 s1B, s2B, s3B, s4B, smaxB, sminB, MSc) = line
                self.eType[eid] = 'CBAR'
                self.s1[eid] = [s1A, s1B]
                self.s2[eid] = [s2A, s2B]
                self.s3[eid] = [s3A, s3B]
                self.s4[eid] = [s4A, s4B]

                self.axial[eid] = axialA
                self.smax[eid] = [smaxA, smaxB]
                self.smin[eid] = [sminA, sminB]
                #self.MS_tension[eid]     = MSt
                #self.MS_compression[eid] = MSc
            return

        (dtName, dt) = transient
        self.data_code['name'] = dtName
        #print "dt = ",dt
        #print "dtName = ",dtName
        if dt not in self.s1:
            self.update_dt(self.data_code, dt)
            self.isTransient = True

        for line in data:
            (eType, eid, s1A, s2A, s3A, s4A, axialA, smaxA, sminA, MSt,
             s1B, s2B, s3B, s4B, smaxB, sminB, MSc) = line
            self.eType[eid] = 'CBAR'
            self.s1[dt][eid] = [s1A, s1B]
            self.s2[dt][eid] = [s2A, s2B]
            self.s3[dt][eid] = [s3A, s3B]
            self.s4[dt][eid] = [s4A, s4B]

            self.axial[dt][eid] = axialA
            self.smax[dt][eid] = [smaxA, smaxB]
            self.smin[dt][eid] = [sminA, sminB]
            #self.MS_tension[dt][eid]     = MSt
            #self.MS_compression[dt][eid] = MSc

    def getLength(self):
        return (68, 'iffffffffffffffff')

    def delete_transient(self, dt):
        del self.s1[dt]
        del self.s2[dt]
        del self.s3[dt]
        del self.s4[dt]
        del self.axial[dt]
        del self.smax[dt]
        del self.smin[dt]

    def get_transients(self):
        k = list(self.s1.keys())
        k.sort()
        return k

    def add_new_transient(self, dt):
        """
        initializes the transient variables
        """
        #print "****add new transient****"
        self.dt = dt
        self.s1[dt] = {}
        self.s2[dt] = {}
        self.s3[dt] = {}
        self.s4[dt] = {}
        self.axial[dt] = {}
        self.smax[dt] = {}
        self.smin[dt] = {}
        #self.MS_tension[dt]     = {}
        #self.MS_compression[dt] = {}

    def addNewEid100(self, dt, out):
        #print "out = ",out
        #return
        (eid, s1, s2, s3, s4, axial, smax, smin, MSt, MSc) = out
        #print "Bar Stress add..."
        self.eType[eid] = 'CBAR'  # eType

        if self.eid in self.s1:
            self.s1[eid].append(s1)
            self.s2[eid].append(s2)
            self.s3[eid].append(s3)
            self.s4[eid].append(s4)
            self.axial[eid].append(axial)
            self.smax[eid].append(smax)
            self.smin[eid].append(smin)
            #self.MS_tension[eid].append(MSt)
            #self.MS_compression[eid].append(MSc)
        else:
            self.s1[eid] = [s1]
            self.s2[eid] = [s2]
            self.s3[eid] = [s3]
            self.s4[eid] = [s4]
            self.axial[eid] = axial
            self.smax[eid] = [smax]
            self.smin[eid] = [smin]
            #self.MS_tension[eid]     = MSt
            #self.MS_compression[eid] = MSc

        #msg = "eid=%s nodeID=%s fd=%g oxx=%g oyy=%g \ntxy=%g angle=%g major=%g minor=%g vm=%g" %(eid,nodeID,fd,oxx,oyy,txy,angle,majorP,minorP,ovm)
        #print msg
        #if nodeID==0: raise Exception(msg)

    def add_new_eid(self, eType, dt, eid, s1a, s2a, s3a, s4a, axial, smaxa, smina, MSt,
        s1b, s2b, s3b, s4b, smaxb, sminb, MSc):
        #print "Bar Stress add..."
        self.eType[eid] = eType

        self.s1[eid] = [s1a, s1b]
        self.s2[eid] = [s2a, s2b]
        self.s3[eid] = [s3a, s3b]
        self.s4[eid] = [s4a, s4b]
        self.axial[eid] = axial
        self.smax[eid] = [smaxa, smaxb]
        self.smin[eid] = [smina, sminb]
        #self.MS_tension[eid]     = MSt
        #self.MS_compression[eid] = MSc

        #msg = "eid=%s nodeID=%s fd=%g oxx=%g oyy=%g \ntxy=%g angle=%g major=%g minor=%g vm=%g" %(eid,nodeID,fd,oxx,oyy,txy,angle,majorP,minorP,ovm)
        #print msg
        #if nodeID==0: raise Exception(msg)

    def add_new_eid_sort1(self, eType, dt, eid, s1a, s2a, s3a, s4a, axial, smaxa, smina, MSt,
                                             s1b, s2b, s3b, s4b, smaxb, sminb, MSc):
        #msg = "dt=%s eid=%s s1a=%s" % (dt, eid, s1a)
        #print msg
        if dt not in self.s1:
            self.add_new_transient(dt)
        self.eType[eid] = eType
        #print self.s1
        self.s1[dt][eid] = [s1a, s1b]
        self.s2[dt][eid] = [s2a, s2b]
        self.s3[dt][eid] = [s3a, s3b]
        self.s4[dt][eid] = [s4a, s4b]
        self.axial[dt][eid] = axial
        self.smax[dt][eid] = [smaxa, smaxb]
        self.smin[dt][eid] = [smina, sminb]
        #self.MS_tension[dt][eid]     = MSt
        #self.MS_compression[dt][eid] = MSc

        #if nodeID==0: raise Exception(msg)

    def write_f06(self, header, pageStamp, pageNum=1, f=None, is_mag_phase=False):
        if self.nonlinear_factor is not None:
            return self._write_f06_transient(header, pageStamp, pageNum, f)

        msg = header + [
                '                                 S T R E S S E S   I N   B A R   E L E M E N T S          ( C B A R )\n',
                '  ELEMENT        SA1            SA2            SA3            SA4           AXIAL          SA-MAX         SA-MIN     M.S.-T\n',
                '    ID.          SB1            SB2            SB3            SB4           STRESS         SB-MAX         SB-MIN     M.S.-C\n',
              ]

        for eid, S1s in sorted(self.s1.items()):
            #eType = self.eType[eid]
            axial = self.axial[eid]
            #MSt = self.MSt[eid]
            #MSc = self.MSc[eid]
            MSt = ''
            MSc = ''

            s1 = self.s1[eid]
            s2 = self.s2[eid]
            s3 = self.s3[eid]
            s4 = self.s4[eid]
            smax = self.smax[eid]
            smin = self.smin[eid]
            vals = [s1[0], s2[0], s3[0], s4[0], axial, smax[0], smin[0],
                    s1[1], s2[1], s3[1], s4[1], smax[1], smin[1]]
            (vals2, isAllZeros) = writeFloats13E(vals)
            [s1a, s2a, s3a, s4a, axial, smaxa, smina,
             s1b, s2b, s3b, s4b, smaxb, sminb] = vals2
            msg.append('0%8i   %13s  %13s  %13s  %13s  %13s  %13s  %13s %-s\n' % (eid, s1a, s2a, s3a, s4a, axial, smaxa, smina, MSt.rstrip()))
            msg.append(' %8s   %13s  %13s  %13s  %13s  %13s  %13s  %13s %-s\n' % ('', s1b, s2b, s3b, s4b, '', smaxb, sminb, MSc.rstrip()))

        msg.append(pageStamp + str(pageNum) + '\n')
        return (''.join(msg), pageNum)

    def _write_f06_transient(self, header, pageStamp, pageNum=1, f=None, is_mag_phase=False):
        words = [
                '                                 S T R E S S E S   I N   B A R   E L E M E N T S          ( C B A R )\n',
                '  ELEMENT        SA1            SA2            SA3            SA4           AXIAL          SA-MAX         SA-MIN     M.S.-T\n',
                '    ID.          SB1            SB2            SB3            SB4           STRESS         SB-MAX         SB-MIN     M.S.-C\n',
              ]
        msg = []
        for dt, S1s in sorted(self.s1.items()):
            header[1] = ' %s = %10.4E\n' % (self.data_code['name'], dt)
            msg += header + words
            for eid, S1 in sorted(S1s.items()):
                #eType = self.eType[eid]
                axial = self.axial[dt][eid]
                #MSt = self.MSt[eid]
                #MSc = self.MSc[eid]
                MSt = ''
                MSc = ''

                s1 = self.s1[dt][eid]
                s2 = self.s2[dt][eid]
                s3 = self.s3[dt][eid]
                s4 = self.s4[dt][eid]
                smax = self.smax[dt][eid]
                smin = self.smin[dt][eid]
                vals = [s1[0], s2[0], s3[0], s4[0], axial, smax[0], smin[0],
                        s1[1], s2[1], s3[1], s4[1], smax[1], smin[1]]
                (vals2, isAllZeros) = writeFloats13E(vals)
                [s1a, s2a, s3a, s4a, axial, smaxa, smina,
                 s1b, s2b, s3b, s4b, smaxb, sminb] = vals2
                msg.append('0%8i   %13s  %13s  %13s  %13s  %13s  %13s  %13s %-s\n' % (eid, s1a, s2a, s3a, s4a, axial, smaxa, smina, MSt.rstrip()))
                msg.append(' %8s   %13s  %13s  %13s  %13s  %13s  %13s  %13s %-s\n' % ('', s1b, s2b, s3b, s4b, '', smaxb, sminb, MSc.rstrip()))

            msg.append(pageStamp + str(pageNum) + '\n')
            pageNum += 1
        return (''.join(msg), pageNum - 1)

    def __repr__(self):
        if self.nonlinear_factor is not None:
            return self.__reprTransient__()

        msg = '---BAR STRESS---\n'
        msg += '%-6s %6s ' % ('EID', 'eType')
        headers = ['s1', 's2', 's3', 's4', 'Axial', 'sMax', 'sMin']
        for header in headers:
            msg += '%8s ' % header
        msg += '\n'

        for eid, S1s in sorted(self.s1.items()):
            eType = self.eType[eid]
            axial = self.axial[eid]
            #MSt = self.MSt[eid]
            #MSc = self.MSc[eid]

            s1 = self.s1[eid]
            s2 = self.s2[eid]
            s3 = self.s3[eid]
            s4 = self.s4[eid]
            smax = self.smax[eid]
            smin = self.smin[eid]
            msg += '%-6i %6s ' % (eid, eType)
            vals = [s1[0], s2[0], s3[0], s4[0], axial, smax[0], smin[0]]
            for val in vals:
                if abs(val) < 1e-6:
                    msg += '%8s ' % '0'
                else:
                    msg += '%8i ' % val
            msg += '\n'

            msg += '%s ' % (' ' * 13)
            vals = [s1[1], s2[1], s3[1], s4[1], '', smax[1], smin[1]]
            for val in vals:
                if is_string(val):
                    msg += '%8s ' % val
                elif abs(val) < 1e-6:
                    msg += '%8s ' % '0'
                else:
                    msg += '%8i ' % val
            msg += '\n'

            #msg += "eid=%-4s eType=%s s1=%-4i s2=%-4i s3=%-4i s4=%-4i axial=-%5i smax=%-5i smax=%-4i\n" %(eid,eType,s1[0],s2[0],s3[0],s4[0],axial, smax[0],smin[0])
            #msg += "%s                s1=%-4i s2=%-4i s3=%-4i s4=%-4i %s         smax=%-5i smax=%-4i\n" %(' '*4,    s1[1],s2[1],s3[1],s4[1],'    ',smax[1],smin[1])

        return msg

    def __reprTransient__(self):
        msg = '---BAR STRESS---\n'
        msg += '%-6s %6s ' % ('EID', 'eType')
        headers = ['s1', 's2', 's3', 's4', 'Axial', 'sMax', 'sMin']
        for header in headers:
            msg += '%8s ' % header
        msg += '\n'

        for dt, S1ss in sorted(self.s1.items()):
            msg += '%s = %g\n' % (self.data_code['name'], dt)
            for eid, S1s in sorted(S1ss.items()):
                eType = self.eType[eid]
                axial = self.axial[dt][eid]
                #MSt = self.MSt[dt][eid]
                #MSc = self.MSc[dt][eid]

                s1 = self.s1[dt][eid]
                s2 = self.s2[dt][eid]
                s3 = self.s3[dt][eid]
                s4 = self.s4[dt][eid]
                smax = self.smax[dt][eid]
                smin = self.smin[dt][eid]
                msg += '%-6i %6s ' % (eid, eType)
                vals = [s1[0], s2[0], s3[0], s4[0], axial, smax[0], smin[0]]
                for val in vals:
                    if abs(val) < 1e-6:
                        msg += '%8s ' % '0'
                    else:
                        msg += '%8i ' % val
                msg += '\n'

                msg += '%s ' % (' ' * 13)
                vals = [s1[1], s2[1], s3[1], s4[1], '', smax[1], smin[1]]
                for val in vals:
                    if is_string(val):
                        msg += '%8s ' % val
                    elif abs(val) < 1e-6:
                        msg += '%8s ' % '0'
                    else:
                        msg += '%8i ' % val
                msg += '\n'

                #msg += "eid=%-4s eType=%s s1=%-4i s2=%-4i s3=%-4i s4=%-4i axial=-%5i smax=%-5i smax=%-4i\n" %(eid,eType,s1[0],s2[0],s3[0],s4[0],axial, smax[0],smin[0])
                #msg += "%s                s1=%-4i s2=%-4i s3=%-4i s4=%-4i %s         smax=%-5i smax=%-4i\n" %(' '*4,    s1[1],s2[1],s3[1],s4[1],'    ',smax[1],smin[1])
        return msg


class BarStrainObject(StrainObject):
    """
    ::

      # s_code=10
                                       S T R A I N S   I N   B A R   E L E M E N T S          ( C B A R )
      ELEMENT        SA1            SA2            SA3            SA4           AXIAL          SA-MAX         SA-MIN     M.S.-T
        ID.          SB1            SB2            SB3            SB4           STRAIN         SB-MAX         SB-MIN     M.S.-C
    """
    def __init__(self, data_code, is_sort1, isubcase, dt=None):
        StrainObject.__init__(self, data_code, isubcase)
        self.eType = {}

        self.code = [self.format_code, self.sort_code, self.s_code]
        if self.code in [[1, 0, 0], [1, 0, 1]]:
            self.e1 = {}
            self.e2 = {}
            self.e3 = {}
            self.e4 = {}
            self.axial = {}
            self.emax = {}
            self.emin = {}
            #self.MS_tension = {}
            #self.MS_compression = {}
        elif self.code == [1, 0, 10]:
            self.e1 = {}
            self.e2 = {}
            self.e3 = {}
            self.e4 = {}
            self.axial = {}
            self.emax = {}
            self.emin = {}
            self.MS_tension = {}
            self.MS_compression = {}
        else:
            raise RuntimeError("Invalid Code: barStrain - get the format/sort/stressCode=%s" % (self.code))

        if is_sort1:
            if dt is not None:
                self.add = self.add_sort1
                self.add_new_eid = self.NewEidSort1
        else:
            assert dt is not None
            self.add = self.addSort2
            self.add_new_eid = self.NewEidSort2

    def get_stats(self):
        nelements = len(self.eType)

        msg = self.get_data_code()
        if self.dt is not None:  # transient
            ntimes = len(self.e1)
            msg.append('  type=%s ntimes=%s nelements=%s\n'
                       % (self.__class__.__name__, ntimes, nelements))
        else:
            msg.append('  type=%s nelements=%s\n' % (self.__class__.__name__,
                                                     nelements))
        msg.append('  eType, e1, e2, e3, e4, axial, emax, emin, '
                   'MS_tension, MS_compression\n')
        return msg

    def add_f06_data(self, data, transient):
        if transient is None:
            for line in data:
                (eType, eid, e1A, e2A, e3A, e4A, axialA, emaxA, eminA, MSt,
                           e1B, e2B, e3B, e4B, emaxB, eminB, MSc) = line
                self.eType[eid] = 'CBAR'
                self.e1[eid] = [e1A, e1B]
                self.e2[eid] = [e2A, e2B]
                self.e3[eid] = [e3A, e3B]
                self.e4[eid] = [e4A, e4B]

                self.axial[eid] = axialA
                self.emax[eid] = [emaxA, emaxB]
                self.emin[eid] = [eminA, eminB]
                #self.MS_tension[eid]     = MSt
                #self.MS_compression[eid] = MSc
            return

        (dtName, dt) = transient
        self.data_code['name'] = dtName
        if dt not in self.s1:
            self.update_dt(self.data_code, dt)
            self.isTransient = True

        for line in data:
            (eType, eid, e1A, e2A, e3A, e4A, axialA, emaxA, eminA, MSt,
                       e1B, e2B, e3B, e4B, emaxB, eminB, MSc) = line
            self.eType[eid] = 'CBAR'
            self.e1[dt][eid] = [e1A, e1B]
            self.e2[dt][eid] = [e2A, e2B]
            self.e3[dt][eid] = [e3A, e3B]
            self.e4[dt][eid] = [e4A, e4B]

            self.axial[dt][eid] = axialA
            self.emax[dt][eid] = [emaxA, emaxB]
            self.emin[dt][eid] = [eminA, eminB]
            #self.MS_tension[dt][eid]     = MSt
            #self.MS_compression[dt][eid] = MSc

    def delete_transient(self, dt):
        del self.e1[dt]
        del self.e2[dt]
        del self.e3[dt]
        del self.e4[dt]
        del self.exial[dt]
        del self.emax[dt]
        del self.emin[dt]

    def get_transients(self):
        k = list(self.e1.keys())
        k.sort()
        return k

    def add_new_transient(self, dt):
        """
        initializes the transient variables
        """
        self.e1[dt] = {}
        self.e2[dt] = {}
        self.e3[dt] = {}
        self.e4[dt] = {}
        self.axial[dt] = {}
        self.emax[dt] = {}
        self.emin[dt] = {}
        #self.MS_tension[dt]     = {}
        #self.MS_compression[dt] = {}

    def add_new_eid(
        self, eType, dt, eid, e1a, e2a, e3a, e4a, axial, emaxa, emina, MSt,
                                    e1b, e2b, e3b, e4b, emaxb, eminb, MSc):
        #print "Bar Stress add..."
        self.eType[eid] = eType
        self.e1[eid] = [e1a, e1b]
        self.e2[eid] = [e2a, e2b]
        self.e3[eid] = [e3a, e3b]
        self.e4[eid] = [e4a, e4b]
        self.axial[eid] = axial
        self.emax[eid] = [emaxa, emaxb]
        self.emin[eid] = [emina, eminb]
        #self.MS_tension[eid]     = MSt
        #self.MS_compression[eid] = MSc

        #msg = "eid=%s nodeID=%s fd=%g oxx=%g oyy=%g \ntxy=%g angle=%g major=%g minor=%g vm=%g" %(eid,nodeID,fd,oxx,oyy,txy,angle,majorP,minorP,ovm)
        #print msg
        #if nodeID==0: raise Exception(msg)

    def add_new_eid_sort1(
        self, eType, dt, eid, e1a, e2a, e3a, e4a, axial, emaxa, emina, MSt,
                                         e1b, e2b, e3b, e4b, emaxb, eminb, MSc):
        #print "Bar Stress add..."

        self.eType[eid] = eType
        if dt not in self.e1:
            self.add_new_transient(dt)

        self.e1[dt][eid] = [e1a, e1b]
        self.e2[dt][eid] = [e2a, e2b]
        self.e3[dt][eid] = [e3a, e3b]
        self.e4[dt][eid] = [e4a, e4b]
        self.axial[dt][eid] = axial
        self.emax[dt][eid] = [emaxa, emaxb]
        self.emin[dt][eid] = [emina, eminb]
        #self.MS_tension[dt][eid]     = MSt
        #self.MS_compression[dt][eid] = MSc

        #msg = "eid=%s nodeID=%s fd=%g oxx=%g oyy=%g \ntxy=%g angle=%g major=%g minor=%g vm=%g" %(eid,nodeID,fd,oxx,oyy,txy,angle,majorP,minorP,ovm)
        #print msg
        #if nodeID==0: raise Exception(msg)

    def write_f06(self, header, pageStamp, pageNum=1, f=None, is_mag_phase=False):
        if self.isTransient:
            return self._write_f06_transient(header, pageStamp, pageNum, f)

        msg = header + [
                '                                  S T R A I N S    I N   B A R   E L E M E N T S          ( C B A R )\n',
                '  ELEMENT        SA1            SA2            SA3            SA4           AXIAL          SA-MAX         SA-MIN     M.S.-T\n',
                '    ID.          SB1            SB2            SB3            SB4           STRAIN         SB-MAX         SB-MIN     M.S.-C\n',
              ]
        for eid, E1s in sorted(self.e1.items()):
            #eType = self.eType[eid]
            axial = self.axial[eid]
            #MSt = self.MSt[eid]
            #MSc = self.MSc[eid]
            MSt = ''
            MSc = ''

            e1 = self.e1[eid]
            e2 = self.e2[eid]
            e3 = self.e3[eid]
            e4 = self.e4[eid]
            emax = self.emax[eid]
            emin = self.emin[eid]
            vals = [e1[0], e2[0], e3[0], e4[0], axial, emax[0], emin[0],
                    e1[1], e2[1], e3[1], e4[1], emax[1], emin[1]]
            (vals2, isAllZeros) = writeFloats13E(vals)
            [e10, e20, e30, e40, axial, emax0, emin0,
             e11, e21, e31, e41, emax1, emin1] = vals2

            msg.append('0%8i   %13s  %13s  %13s  %13s  %13s  %13s  %13s %-s\n' % (eid, e10, e20, e30, e40, axial, emax0, emin0, MSt.rstrip()))
            msg.append(' %8s   %13s  %13s  %13s  %13s  %13s  %13s  %13s %-s\n' % ('', e11, e21, e31, e41, '', emax1, emin1, MSc.rstrip()))

        msg.append(pageStamp + str(pageNum) + '\n')
        return (''.join(msg), pageNum)

    def _write_f06_transient(self, header, pageStamp, pageNum=1, f=None, is_mag_phase=False):
        words = [
                '                                  S T R A I N S    I N   B A R   E L E M E N T S           ( C B A R )\n',
                '  ELEMENT        SA1            SA2            SA3            SA4           AXIAL          SA-MAX         SA-MIN     M.S.-T\n',
                '    ID.          SB1            SB2            SB3            SB4           STRAIN         SB-MAX         SB-MIN     M.S.-C\n',
              ]
        msg = []
        for dt, E1s in sorted(self.e1.items()):
            header[1] = ' %s = %10.4E\n' % (self.data_code['name'], dt)
            msg += header + words
            for eid, e1s in sorted(E1s.items()):
                #eType = self.eType[eid]
                axial = self.axial[eid]
                #MSt = self.MSt[eid]
                #MSc = self.MSc[eid]
                MSt = ''
                MSc = ''

                e1 = self.e1[eid]
                e2 = self.e2[eid]
                e3 = self.e3[eid]
                e4 = self.e4[eid]
                emax = self.emax[eid]
                emin = self.emin[eid]
                vals = [e1[0], e2[0], e3[0], e4[0], axial, emax[0], emin[0],
                        e1[1], e2[1], e3[1], e4[1], emax[1], emin[1]]
                (vals2, isAllZeros) = writeFloats13E(vals)
                [e10, e20, e30, e40, axial, emax0, emin0,
                 e11, e21, e31, e41, emax1, emin1] = vals2

                msg.append('0%8i   %13s  %13s  %13s  %13s  %13s  %13s  %13s %-s\n' % (eid, e10, e20, e30, e40, axial, emax0, emin0, MSt.rstrip()))
                msg.append(' %8s   %13s  %13s  %13s  %13s  %13s  %13s  %13s %-s\n' % ('', e11, e21, e31, e41, '', emax1, emin1, MSc.rstrip()))
            msg.append(pageStamp + str(pageNum) + '\n')
            pageNum += 1
        return (''.join(msg), pageNum - 1)

    def __repr__(self):
        if self.isTransient:
            return self.__reprTransient__()

        msg = '---BAR STRAIN---\n'
        msg += '%-8s %6s ' % ('EID', 'eType')
        headers = ['e1', 'e2', 'e3', 'e4', 'Axial', 'eMax', 'eMin']
        for header in headers:
            msg += '%10s ' % header
        msg += '\n'

        for eid, E1s in sorted(self.e1.items()):
            eType = self.eType[eid]
            axial = self.axial[eid]
            #MSt  = self.MS_tension[eid]
            #MSc  = self.MS_compression[eid]

            e1 = self.e1[eid]
            e2 = self.e2[eid]
            e3 = self.e3[eid]
            e4 = self.e4[eid]
            emax = self.emax[eid]
            emin = self.emin[eid]
            msg += '%-8i %6s ' % (eid, eType)
            vals = [e1[0], e2[0], e3[0], e4[0], axial, emax[0], emin[0]]
            for val in vals:
                if abs(val) < 1e-6:
                    msg += '%10s ' % '0'
                else:
                    msg += '%10.3g ' % val
            msg += '\n'

            msg += '%s ' % (' ' * 17)
            vals = [e1[1], e2[1], e3[1], e4[1], '', emax[1], emin[1]]
            for val in vals:
                if is_string(val):
                    msg += '%10s ' % val
                elif abs(val) < 1e-6:
                    msg += '%10s ' % '0'
                else:
                    msg += '%10.3g ' % val
            msg += '\n'

            #msg += "eid=%-4s eType=%s s1=%-4i s2=%-4i s3=%-4i s4=%-4i axial=-%5i smax=%-5i smax=%-4i\n" %(eid,eType,s1[0],s2[0],s3[0],s4[0],axial, smax[0],smin[0])
            #msg += "%s                s1=%-4i s2=%-4i s3=%-4i s4=%-4i %s         smax=%-5i smax=%-4i\n" %(' '*4,    s1[1],s2[1],s3[1],s4[1],'    ',smax[1],smin[1])

        return msg

    def __reprTransient__(self):
        msg = '---BAR STRAIN---\n'
        msg += '%-8s %6s ' % ('EID', 'eType')
        headers = ['e1', 'e2', 'e3', 'e4', 'Axial', 'eMax', 'eMin']
        for header in headers:
            msg += '%10s ' % header
        msg += '\n'

        for dt, E1s in sorted(self.e1.items()):
            msg += "%s = %g\n" % (self.data_code['name'], dt)
            for eid, e1s in sorted(Els.items()):
                eType = self.eType[eid]
                axial = self.axial[dt][eid]
                #MSt  = self.MS_tension[dt][eid]
                #MSc  = self.MS_compression[dt][eid]

                e1 = self.e1[dt][eid]
                e2 = self.e2[dt][eid]
                e3 = self.e3[dt][eid]
                e4 = self.e4[dt][eid]
                emax = self.emax[dt][eid]
                emin = self.emin[dt][eid]
                msg += '%-8i %6s ' % (eid, eType)
                vals = [e1[0], e2[0], e3[0], e4[0], axial, emax[0], emin[0]]
                for val in vals:
                    if abs(val) < 1e-6:
                        msg += '%10s ' % '0'
                    else:
                        msg += '%10.3g ' % val
                msg += '\n'

                msg += '%s ' % (' ' * 17)
                vals = [e1[1], e2[1], e3[1], e4[1], '', emax[1], emin[1]]
                for val in vals:
                    if is_string(val):
                        msg += '%10s ' % val
                    elif abs(val) < 1e-6:
                        msg += '%10s ' % '0'
                    else:
                        msg += '%10.3g ' % val
                msg += '\n'

                #msg += "eid=%-4s eType=%s s1=%-4i s2=%-4i s3=%-4i s4=%-4i axial=-%5i smax=%-5i smax=%-4i\n" %(eid,eType,s1[0],s2[0],s3[0],s4[0],axial, smax[0],smin[0])
                #msg += "%s                s1=%-4i s2=%-4i s3=%-4i s4=%-4i %s         smax=%-5i smax=%-4i\n" %(' '*4,    s1[1],s2[1],s3[1],s4[1],'    ',smax[1],smin[1])

        return msg
