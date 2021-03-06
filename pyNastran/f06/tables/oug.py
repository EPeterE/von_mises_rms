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
#pylint: disable=E1101,W0612,R0201


from pyNastran.op2.tables.oug.oug_displacements import DisplacementObject, ComplexDisplacementObject
from pyNastran.op2.tables.oug.oug_temperatures import TemperatureObject


class OUG(object):
    def __init__(self):
        self.displacements = {}
        self.temperatures = {}

    def getDisplacement(self):
        """
        ::
                                               D I S P L A C E M E N T   V E C T O R
  
          POINT ID.   TYPE          T1             T2             T3             R1             R2             R3
                 1      G      9.663032E-05   0.0           -2.199001E-04   0.0           -9.121119E-05   0.0
                 2      G      0.0            0.0            0.0            0.0            0.0            0.0
                 3      G      0.0            0.0            0.0            0.0            0.0            0.0

        * analysis_code = 1 (Statics)
        * device_code   = 1 (Print)
        * table_code    = 1 (Displacement)
        * sort_code     = 0 (Sort2,Real,Sorted Results) => sort_bits = [0,0,0]
        * num_wide      = 8 (???)
        """
        (subcaseName, isubcase, transient, dt, analysis_code,
            is_sort1) = self.readSubcaseNameID()
        headers = self.skip(2)
        data_code = {'log': self.log, 'analysis_code': analysis_code,
                    'device_code': 1, 'table_code': 1,
                    'sort_code': 0, 'sort_bits': [0, 0, 0], 'num_wide': 8,
                    'table_name': 'OUG', 'nonlinear_factor': dt}
        #print "headers = %s" %(headers)
        dataTypes = [int, str, float, float, float, float, float, float]
        data = self.readTable(dataTypes)

        if isubcase in self.displacements:
            self.displacements[isubcase].add_f06_data(data, transient)
        else:
            is_sort1 = True
            disp = DisplacementObject(data_code, is_sort1, isubcase)
            disp.add_f06_data(data, transient)
            self.displacements[isubcase] = disp
        self.iSubcases.append(isubcase)

    def getComplexDisplacement(self):
        """
        ::

            BACKWARD WHIRL
                                                                                                                   SUBCASE 2
            POINT-ID =       101
                                             C O M P L E X   D I S P L A C E M E N T   V E C T O R
                                                               (MAGNITUDE/PHASE)
  
            FREQUENCY   TYPE          T1             T2             T3             R1             R2             R3
          2.000000E+01     G      3.242295E-16   1.630439E-01   1.630439E-01   1.691497E-17   1.362718E-01   1.362718E-01
                                  196.0668        90.0000       180.0000        63.4349       180.0000       270.0000

        * table_code    = 1 (Displacement)
        * format_code   = 3 (Magnitude/Phase)
        * sort_bits     = [0,1,1]  (Sort1,Real/Imaginary,RandomResponse)
        * analysis_code = 5 (Frequency)
        * sort_code     = 2 (Random Response)
        """
        (subcaseName, isubcase, transient, dt, analysis_code,
            is_sort1) = self.readSubcaseNameID()
        headers = self.skip(3)
        data = []

        data_code = {'log': self.log, 'analysis_code': 5, 'device_code': 1,
                    'table_code': 1, 'sort_code': 2, 'sort_bits': [0, 1, 1],
                    'num_wide': 14, 'format_code': 3, 'table_name': 'OUGV1',
                    'nonlinear_factor': dt,
                    #'mode':iMode,'eigr':transient[1], 'mode_cycle':cycle,
                    #'dataNames':['mode', 'eigr', 'mode_cycle'],
                    #'name':'mode',
                    #'s_code':0,
                    #'element_name':'CBAR', 'element_type':34, 'stress_bits':stress_bits,
                    }
        while 1:
            line = self.infile.readline()[1:].rstrip('\r\n ')
            if 'PAGE' in line:
                break
            sline = line.strip().split()
            line2 = self.infile.readline()[1:].rstrip('\r\n ')
            sline += line2.strip().split()
            
            dx = float(sline[2]) + float(sline[8])*1j
            dy = float(sline[3]) + float(sline[9])*1j
            dz = float(sline[4]) + float(sline[10])*1j
            rx = float(sline[5]) + float(sline[11])*1j
            ry = float(sline[6]) + float(sline[12])*1j
            rz = float(sline[7]) + float(sline[13])*1j
            out = [float(sline[0]), sline[1].strip(), dx, dy, dz, rx, ry, rz]
            #print sline
            #print out
            data.append(out)
            self.i += 2
        self.i += 1

        if isubcase in self.displacements:
            self.displacements[isubcase].add_f06_data(data, transient)
        else:
            is_sort1 = True
            disp = ComplexDisplacementObject(data_code, is_sort1, isubcase)
            disp.add_f06_data(data, transient)
            self.displacements[isubcase] = disp
        self.iSubcases.append(isubcase)

    def getTemperatureVector(self):
        """
        @code
        LOAD STEP =  1.00000E+00
                                              T E M P E R A T U R E   V E C T O R

        POINT ID.   TYPE      ID   VALUE     ID+1 VALUE     ID+2 VALUE     ID+3 VALUE     ID+4 VALUE     ID+5 VALUE
               1      S      1.300000E+03   1.300000E+03   1.300000E+03   1.300000E+03   1.300000E+03   1.300000E+03
               7      S      1.300000E+03   1.300000E+03   1.300000E+03   1.300000E+03
        analysis_code = 1 (Statics)
        device_code   = 1 (Print)
        table_code    = 1 (Displacement/Temperature)
        sort_code     = 0 (Sort2,Real,Sorted Results) => sort_bits = [0,0,0]
        format_code   = 1 (Real)
        s_code        = 0 (Stress)
        num_wide      = 8 (???)
        @endcode
        """
        (subcaseName, isubcase, transient, dt, analysis_code,
            is_sort1) = self.readSubcaseNameID()
        #print transient

        headers = self.skip(2)
        #print "headers = %s" %(headers)
        data = self.readTemperatureTable()

        data_code = {'log': self.log, 'analysis_code': 1, 'device_code': 1,
                    'table_code': 1, 'sort_code': 0, 'sort_bits': [0, 0, 0],
                    'num_wide': 8, 'table_name': 'OUG', 'nonlinear_factor': dt,
                    #'thermalCode':1,
                    #'format_code':1,
                    #'element_name':eType,'s_code':0,'stress_bits':stress_bits
                    }

        if isubcase in self.temperatures:
            self.temperatures[isubcase].add_f06_data(data, transient)
        else:
            is_sort1 = True
            temp = TemperatureObject(data_code, is_sort1, isubcase)
            temp.add_f06_data(data, transient)
            self.temperatures[isubcase] = temp
        self.iSubcases.append(isubcase)

    def readTemperatureTable(self):
        data = []
        Format = [int, str, float, float, float, float, float, float]
        while 1:
            line = self.infile.readline()[1:].rstrip('\r\n ')
            if 'PAGE' in line:
                return data
            sline = [line[0:15], line[15:22].strip(), line[22:40], line[40:55], line[55:70], line[70:85], line[85:100], line[100:115]]
            sline = self.parseLineTemperature(sline, Format)
            data.append(sline)
        return data

    def parseLineTemperature(self, sline, Format):
        out = []
        for (entry, iFormat) in zip(sline, Format):
            if entry is '':
                return out
            #print "sline=|%r|\n entry=|%r| format=%r" %(sline,entry,iFormat)
            entry2 = iFormat(entry)
            out.append(entry2)
        return out
