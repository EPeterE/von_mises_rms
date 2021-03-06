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
# pylint: disable=E1101,C0103,R0902,R0904,R0914



from numpy import array

from pyNastran.op2.resultObjects.op2_Objects import scalarObject


class nonlinearFluxObject(scalarObject):  # approach_code=10, sort_code=0
    def __init__(self, data_code, isubcase, load_step):
        scalarObject.__init__(self, data_code, isubcase)

        self.load_step = load_step
        self.eTypes = {}
        self.fluxes = {}
        self.gradients = {}
        if load_step is not None:
            self.add_new_transient()
            #self.isTransient = True
            #raise Exception('transient not supported for flux yet...')

    def update_dt(self, data_code, load_step):
        self.data_code = data_code
        self.apply_data_code()
        assert load_step >= 0.
        self.load_step = load_step
        self.add_new_transient()

    def add_new_transient(self):
        """
        initializes the transient variables
        .. note:: make sure you set self.dt first
        """
        self.fluxes[self.load_step] = {}
        self.gradients[self.load_step] = {}

    def add(self, nodeID, eType, v1, v2, v3, v4=None, v5=None, v6=None):
        assert 0 < nodeID < 1000000000, 'nodeID=%s' % (nodeID)
        #print("nodeID=%s eType=%s v1=%s v2=%s v3=%s v4=%s v5=%s v6=%s"
        #    %(nodeID,eType,v1,v2,v3,v4,v5,v6))
        assert nodeID not in self.fluxes[self.load_step], 'nodeID=%s' % (nodeID)
        self.gradients[self.load_step][nodeID] = array([v1, v2, v3])
        self.fluxes[self.load_step][nodeID] = array([v4, v5, v6])
        self.eTypes[nodeID] = eType

    def __repr__(self):
        msg = '---NONLINEAR GRADIENTS & HEAT FLUX---\n'
        msg += 'load_step = %g\n' % (self.load_step)

        for (dt, fluxPack) in sorted(self.fluxes.items()):
            msg += ('%-10s %-8s %-10s %-10s %-10s %-10s %-10s %-10s\n'
                    % ('GRID', 'eType', 'xGrad', 'yGrad', 'zGrad',
                       'xFlux', 'yFlux', 'zFlux'))

            for (nodeID, flux) in sorted(fluxPack.items()):
                eType = self.eTypes[nodeID]
                msg += '%-10i %-8s ' % (nodeID, eType)
                gradients = self.gradients[dt][nodeID]

                for val in list(gradients) + list(flux):
                    if abs(val) < 1e-6:
                        msg += '%-10s ' % '0.'
                    else:
                        msg += '%-10i ' % val
                msg += '\n'
        return msg
