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
"""
Links up the various cards in the BDF.
"""
# pylint: disable=E1101,C0103,R0902,R0904,R0914,W0611

from __future__ import print_function
import warnings
#from pyNastran.bdf.cards.constraints import constraintObject2


class XrefMesh(object):
    """
    Links up the various cards in the BDF.
    """
    def __init__(self):
        """
        The main BDF class defines all the parameters that are used.
        """
        pass

    def crossReference(self, xref=True):
        """
        .. deprecated: will be replaced in version 0.7 with :func:`cross_reference`
        """
        warnings.warn('crossReference is deprecated; use cross_reference')
        self.cross_reference(xref)

    def cross_reference(self, xref=True):
        """
        Links up all the cards to the cards they reference
        """
        if xref:
            self.log.debug("Cross Referencing...")
            #for key,e in self.elements.iteritems():
                #print(e)

            self._cross_reference_nodes()
            self._cross_reference_coordinates()

            self._cross_reference_elements()
            self._cross_reference_properties()
            self._cross_reference_materials()

            self._cross_reference_aero()
            self._cross_reference_constraints()
            self._cross_reference_loads()
            #self.caseControlDeck.cross_reference(self)

    def _cross_reference_constraints(self):
        """
        Links the SPCADD, SPC, SPCAX, SPCD, MPCADD, MPC cards.
        """
        #self.spcObject.cross_reference(self)  # enable to output SPCs
        #self.mpcObject.cross_reference(self)  # enable to output MPCs

        #self.spcObject = constraintObject()
        for spcadd in self.spcadds.values():
            self.spcObject.Add(spcadd)

        for spcs in self.spcs.values():
            for spc in spcs:
                self.spcObject.append(spc)

        for mpcadd in self.mpcadds.values():
            self.mpcObject.Add(mpcadd)

        for mpcs in self.mpcs.values():
            for mpc in mpcs:
                self.mpcObject.append(mpc)
        #self.mpcObject = constraintObject()
        #self.spcObject.cross_reference(self)

    def _cross_reference_coordinates(self):
        """
        Links up all the coordinate cards to other coordinate cards and nodes
        """
        # CORD2x: links the rid to coordinate systems
        # CORD1x: links g1,g2,g3 to grid points
        for coord in self.coords.values():
            coord.cross_reference(self)

        # CORD1x: Since the grid points were already referenced,
        # we can now resolve the coordinate systems.
        # We couldnt do it in the previous step b/c
        # the grid's coordinate system might have been
        # unresolved
        for coord in self.coords.values():
            coord.resolveCid()

    def _cross_reference_aero(self):
        """
        Links up all the aero cards
        """
        for caero in self.caeros.values():
            caero.cross_reference(self)

        for spline in self.splines.values():
            spline.cross_reference(self)

    def _cross_reference_nodes(self):
        """
        Links the nodes to coordinate systems
        """
        gridSet = self.gridSet
        for n in self.nodes.values():
            try:
                n.cross_reference(self, gridSet)
            except:
                self.log.error("Couldn't cross reference GRID.\n%s" % (str(n)))
                raise

        if self.spoints:
            self.spointi = self.spoints.createSPOINTi()

    def _cross_reference_elements(self):
        """
        Links the elements to nodes, properties (and materials depending on
        the card).
        """
        for elem in self.elements.values():
            try:
                elem.cross_reference(self)
            except:
                msg = "Couldn't cross reference Element.\n%s" % (str(elem))
                self.log.error(msg)
                raise

    def _cross_reference_properties(self):
        """
        Links the properties to materials
        """
        for prop in self.properties.values():
            try:
                prop.cross_reference(self)
            except:
                msg = "Couldn't cross reference Property.\n%s" % (str(prop))
                self.log.error(msg)
                raise

    def _cross_reference_materials(self):
        """
        Links the materials to materials (e.g. MAT1, CREEP)
        often this is a pass statement
        """
        for mat in self.materials.values():  # MAT1
            try:
                mat.cross_reference(self)
            except:
                msg = "Couldn't cross reference Material\n%s" % (str(mat))
                self.log.error(msg)
                raise

        for mat in self.materialDeps.values():  # CREEP - depends on MAT1
            try:
                mat.cross_reference(self)
            except:
                msg = "Couldn't cross reference Material\n%s" % (str(mat))
                self.log.error(msg)
                raise

    def _cross_reference_loads(self):
        """
        Links the loads to nodes, coordinate systems, and other loads.
        """
        for (lid, sid) in self.loads.items():
            #self.log.debug("lid=%s sid=%s" %(lid,sid))
            for load in sid:
                try:
                    load.cross_reference(self)
                except:
                    self.log.error("lid=%s sid=%s" % (lid, sid))
                    msg = "Couldn't cross reference Load\n%s" % (str(load))
                    self.log.error(msg)
                    raise
