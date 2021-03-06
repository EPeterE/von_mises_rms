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
# pylint: disable=C0103,C0111,W0612,R0912,R0914,R0904,W0613,E1101



import warnings
from pyNastran.bdf.fieldWriter import print_card
#from pyNastran.bdf.bdfInterface.bdf_Reader import print_filename


class WriteMeshDeprecated(object):
    def writeBDF(self, outFileName='fem.out.bdf', size=8, debug=False):
        """
        .. seealso:: write_bdf
        .. deprecated:: will be replaced in version 0.7 with write_bdf with interspersed=False
        """
        warnings.warn('writeBDF has been deprecated; use '
                      'write_bdf', DeprecationWarning, stacklevel=2)
        self.write_bdf(outFileName, size, debug)

    def writeBDFAsPatran(self, outFileName='fem.out.bdf', size=8, debug=False):
        """
        .. seealso:: write_bdf
        .. deprecated:: will be replaced in version 0.7 with write_bdf with an interspersed=True
        """
        warnings.warn('writeBDFAsPatran has been deprecated; use '
                      'write_bdf_as_patran', DeprecationWarning, stacklevel=2)
        self.write_bdf_as_patran(outFileName, size, debug)

    def echoBDF(self, infileName):
        """
        .. seealso:: echo_bdf
        .. deprecated:: will be replaced in version 0.7 with echo_bdf
        """
        warnings.warn('echoBDF has been deprecated; use '
                      'echo_bdf', DeprecationWarning, stacklevel=2)
        self.echo_bdf(infileName)


class WriteMesh(WriteMeshDeprecated):
    def __init__(self):
        pass

    def echo_bdf(self, infile_name):
        """
        This method removes all comment lines from the bdf
        A write method is stil required.

        .. todo:: maybe add the write method
        """
        self.cardsToRead = set([])
        return self.read_bdf(infile_name)

    def auto_reject_bdf(self, infile_name):
        """
        This method parses supported cards, but does not group them into
        nodes, elements, properties, etc.

        .. todo:: maybe add the write method
        """
        self._auto_reject = True
        return self.read_bdf(infile_name)

    def _write_elements_as_CTRIA3(self, size):
        """
        Takes the cquad4 elements and splits them

        :returns msg:  string representation of the elements
        """
        eids = self.elementIDs()
        #print "eids = ",eids
        nextEID = max(eids) + 1  # set the new ID
        msg = '$ELEMENTS\n'
        for eid, element in sorted(self.elements.items()):
            if element.Is('CQUAD4'):
                msg += element.writeAsCTRIA3(nextEID)
                nextEID += 1
            else:
                msg += element.print_card(size)
        return msg

    def _write_dmigs(self, size):
        """
        :param self:  the BDF object
        :param size:  large field (16) or small field (8)
        :returns msg: string representation of the DMIGs
        """
        msg = []
        for (name, dmig) in sorted(self.dmigs.items()):
            msg.append(str(dmig))
        for (name, dmi) in sorted(self.dmis.items()):
            msg.append(str(dmi))
        for (name, dmij) in sorted(self.dmijs.items()):
            msg.append(str(dmij))
        for (name, dmiji) in sorted(self.dmijis.items()):
            msg.append(str(dmiji))
        for (name, dmik) in sorted(self.dmiks.items()):
            msg.append(str(dmik))
        return ''.join(msg)

    def _write_common(self, size):
        """
        method to write the common outputs so none get missed...
        :param self: the BDF object
        :returns msg: part of the bdf
        """
        msg = ''
        msg += self._write_rigid_elements(size)
        msg += self._write_dmigs(size)
        msg += self._write_loads(size)
        msg += self._write_dynamic(size)
        msg += self._write_aero(size)
        msg += self._write_aero_control(size)
        msg += self._write_flutter(size)
        msg += self._write_thermal(size)
        msg += self._write_thermal_materials(size)

        msg += self._write_constraints(size)
        msg += self._write_optimization(size)
        msg += self._write_tables(size)
        msg += self._write_sets(size)
        msg += self._write_rejects(size)
        msg += self._write_coords(size)
        return msg

    def write_bdf(self, out_filename='fem.out.bdf', interspersed=True,
                  size=8, debug=False):
        """
        Writes the BDF.

        :param self:         the BDF object
        :param out_filename: the name to call the output bdf
        :param debug:        developer debug (unused)
        :param interspersed: Writes a bdf with properties & elements
              interspersed like how Patran writes the bdf.  This takes
              slightly longer than if interspersed=False, but makes it
              much easier to compare to a Patran-formatted bdf and is
              more clear. (default=True)
        :param size:  the field size (8 is recommended)
        :param debug: developer debug
        """
        assert isinstance(interspersed, bool)
        assert size in [8, 16]
        #size = 16
        fname = self.print_filename(out_filename)
        self.log.debug("***writing %s" % fname)

        outfile = open(out_filename, 'w')
        msg = self._write_header()
        msg += self._write_params(size)
        outfile.write(msg)

        msg = self._write_nodes(size)
        outfile.write(msg)

        if interspersed:
            msg = self._write_elements_properties(size)
        else:
            msg = self._write_elements(size)
            outfile.write(msg)
            msg = self._write_properties(size)

        outfile.write(msg)

        msg = self._write_materials(size)
        msg += self._write_common(size)
        msg += 'ENDDATA\n'
        outfile.write(msg)
        outfile.close()

    def write_as_CTRIA3(self, out_filename='fem.out.bdf', size=8, debug=False):
        """
        Writes a series of CQUAD4s as CTRIA3s.  All other cards are echoed.
        :param self:         the BDF object
        :param out_filename: the name to call the output bdf
        :param debug:        developer debug (unused)
        .. warning:: not tested in a long time
        """
        assert size in [8, 16]
        #size = 16
        fname = self.print_filename(out_filename)
        self.log.debug("***writing %s" % fname)

        outfile = open(out_filename, 'w')
        msg = self._write_header()
        msg += self._write_params(size)
        outfile.write(msg)

        msg = self._write_nodes(size)
        outfile.write(msg)

        msg = self._write_elements_as_CTRIA3(size)
        outfile.write(msg)

        msg = self._write_properties(size)
        msg += self._write_materials(size)
        msg += self._write_common(size)
        msg += 'ENDDATA\n'
        outfile.write(msg)
        outfile.close()

    def _write_header(self):
        """
        Writes the executive and case control decks.
        :param self: the BDF object
        """
        msg = self._write_executive_control_deck()
        msg += self._write_case_control_deck()
        return msg

    def _write_executive_control_deck(self):
        """
        Writes the executive control deck.
        :param self: the BDF object
        """
        msg = ''
        if self.executive_control_lines:
            msg = '$EXECUTIVE CONTROL DECK\n'
            if self.sol == 600:
                newSol = 'SOL 600,%s' % (self.solMethod)
            else:
                newSol = 'SOL %s' % (self.sol)

            if self.iSolLine is not None:
                self.executive_control_lines[self.iSolLine] = newSol

            for line in self.executive_control_lines:
                msg += line + '\n'
        return msg

    def _write_case_control_deck(self):
        """
        Writes the Case Control Deck.
        :param self: the BDF object
        """
        msg = ''
        if self.caseControlDeck:
            msg += '$CASE CONTROL DECK\n'
            msg += str(self.caseControlDeck)
            assert 'BEGIN BULK' in msg, msg
        return msg

    def _write_params(self, size):
        """
        Writes the PARAM cards
        :param self: the BDF object
        """
        msg = []
        if self.params:
            msg = ['$PARAMS\n']
            for (key, param) in sorted(self.params.items()):
                msg.append(param.print_card(size))
        return ''.join(msg)

    def _write_nodes(self, size):
        """
        Writes the NODE-type cards
        :param self: the BDF object
        """
        msg = []
        if self.spoints:
            msg.append('$SPOINTS\n')
            msg.append(str(self.spoints))

        if self.nodes:
            msg.append('$NODES\n')
            if self.gridSet:
                msg.append(self.gridSet.print_card(size))
            for (nid, node) in sorted(self.nodes.items()):
                msg.append(node.print_card(size))
        if 0:
            self._write_nodes_associated(size)

        return ''.join(msg)

    def _write_nodes_associated(self, size):
        """
        Writes the NODE-type in associated and unassociated groups.
        :param self: the BDF object
        .. warning:: Sometimes crashes, probably on invalid BDFs.
        """
        msg = []
        associated_nodes = set([])
        for (eid, element) in self.elements.items():
            #print(element)
            associated_nodes = associated_nodes.union(set(element.nodeIDs()))

        all_nodes = set(self.nodes.keys())
        unassociated_nodes = list(all_nodes.difference(associated_nodes))
        #missing_nodes = all_nodes.difference(
        associated_nodes = list(associated_nodes)

        if associated_nodes:
            msg += ['$ASSOCIATED NODES\n']
            if self.gridSet:
                msg.append(str(self.gridSet))
            for key, node in sorted(associated_nodes.items()):
                msg.append(node.print_card(size))

        if unassociated_nodes:
            msg.append('$UNASSOCIATED NODES\n')
            if self.gridSet and not associated_nodes:
                msg.append(str(self.gridSet))
            for key, node in sorted(unassociated_nodes.items()):
                if key in self.nodes:
                    msg.append(node.print_card(size))
                else:
                    msg.append('$ Missing NodeID=%s' % key)
        return ''.join(msg)

    def _write_elements(self, size):
        """
        Writes the elements in a sorted order
        :param self: the BDF object
        """
        msg = []
        if self.elements:
            msg = ['$ELEMENTS\n']
            for (eid, element) in sorted(self.elements.items()):
                try:
                    msg.append(element.print_card(size))
                except:
                    print('failed printing element...'
                          'type=%s eid=%s' % (element.type, eid))
                    raise
        return ''.join(msg)

    def _write_rigid_elements(self, size):
        """Writes the rigid elements in a sorted order"""
        msg = []
        if self.rigidElements:
            msg = ['$RIGID ELEMENTS\n']
            for (eid, element) in sorted(self.rigidElements.items()):
                try:
                    msg.append(element.print_card(size))
                except:
                    print('failed printing element...'
                          'type=%s eid=%s' % (element.type, eid))
                    raise
        return ''.join(msg)

    def _write_properties(self, size):
        """Writes the properties in a sorted order"""
        msg = []
        if self.properties:
            msg += ['$PROPERTIES\n']
            for (pid, prop) in sorted(self.properties.items()):
                msg.append(prop.print_card(size))
        return ''.join(msg)

    def _write_elements_properties(self, size):
        """Writes the elements and properties in and interspersed order"""
        msg = []
        missing_properties = []
        if self.properties:
            msg.append('$ELEMENTS_WITH_PROPERTIES\n')

        eids_written = []
        for (pid, prop) in sorted(self.properties.items()):
            eids = self.getElementIDsWithPID(pid)

            if eids:
                msg.append(prop.print_card(size))
                eids.sort()
                for eid in eids:
                    element = self.Element(eid)
                    try:
                        msg.append(element.print_card(size))
                    except:
                        print('failed printing element...'
                              'type=%s eid=%s' % (element.type, eid))
                        raise
                eids_written += eids
            else:
                missing_properties.append(str(prop))

        eids_missing = set(self.elements.keys()).difference(set(eids_written))

        if eids_missing:
            msg.append('$ELEMENTS_WITH_NO_PROPERTIES '
                       '(PID=0 and unanalyzed properties)\n')
            for eid in sorted(eids_missing):
                element = self.Element(eid)
                try:
                    msg.append(element.print_card(size))
                except:
                    print('failed printing element...'
                          'type=%s eid=%s' % (element.type, eid))
                    raise

        if missing_properties or self.pdampt or self.pbusht or self.pelast:
            msg.append('$UNASSOCIATED_PROPERTIES\n')
            for pbusht in sorted(self.pbusht.values()):
                msg.append(str(pbusht))
            for pdampt in sorted(self.pdampt.values()):
                msg.append(str(pdampt))
            for pelast in sorted(self.pelast.values()):
                msg.append(str(pelast))
            for missing_property in missing_properties:
                #print("missing_property = ",missing_property)
                #msg.append(missing_property.print_card(size))
                msg.append(missing_property)
        return ''.join(msg)

    def _write_materials(self, size):
        """Writes the materials in a sorted order"""
        msg = []
        if self.materials:
            msg.append('$MATERIALS\n')
            for (mid, material) in sorted(self.materials.items()):
                msg.append(material.print_card(size))
            for (mid, material) in sorted(self.creepMaterials.items()):
                msg.append(material.print_card(size))
            for (mid, material) in sorted(self.materialDeps.items()):
                msg.append(material.print_card(size))
        return ''.join(msg)

    def _write_thermal_materials(self, size):
        """Writes the thermal materials in a sorted order"""
        msg = []
        if self.thermalMaterials:
            msg.append('$THERMAL MATERIALS\n')
            for (mid, material) in sorted(self.thermalMaterials.items()):
                msg.append(material.print_card(size))
        return ''.join(msg)

    def _write_constraints(self, size):
        """Writes the constraint cards sorted by ID"""
        msg = []
        if self.suports:
            msg.append('$CONSTRAINTS\n')
            for suport in self.suports:
                msg.append(str(suport))

        if self.spcs or self.spcadds:
            msg.append('$SPCs\n')
            strSPC = str(self.spcObject)
            if strSPC:
                msg.append(strSPC)
            else:
                for (spcID, spcadd) in sorted(self.spcadds.items()):
                    msg.append(str(spcadd))
                for (spcID, spcs) in sorted(self.spcs.items()):
                    for spc in spcs:
                        msg.append(str(spc))

        if self.mpcs or self.mpcadds:
            msg.append('$MPCs\n')
            strMPC = str(self.mpcObject)
            if strMPC:
                msg.append(strMPC)
            else:
                for (mpcID, mpcadd) in sorted(self.mpcadds.items()):
                    msg.append(str(mpcadd))
                for (mpcID, mpcs) in sorted(self.mpcs.items()):
                    for mpc in mpcs:
                        msg.append(str(mpc))
        return ''.join(msg)

    def _write_loads(self, size):
        """Writes the load cards sorted by ID"""
        msg = []
        if self.loads:
            msg.append('$LOADS\n')
            for (key, loadcase) in sorted(self.loads.items()):
                for load in loadcase:
                    try:
                        msg.append(load.print_card(size))
                    except:
                        print('failed printing load...type=%s key=%s'
                              % (load.type, key))
                        raise
        return ''.join(msg)

    def _write_optimization(self, size):
        """Writes the optimization cards sorted by ID"""
        msg = []
        if (self.dconstrs or self.desvars or self.ddvals or self.dresps
            or self.dvprels or self.dvmrels or self.doptprm or self.dlinks
            or self.ddvals):
            msg.append('$OPTIMIZATION\n')
            for (ID, dconstr) in sorted(self.dconstrs.items()):
                msg.append(dconstr.print_card(size))
            for (ID, desvar) in sorted(self.desvars.items()):
                msg.append(desvar.print_card(size))
            for (ID, ddval) in sorted(self.ddvals.items()):
                msg.append(ddval.print_card(size))
            for (ID, dlink) in sorted(self.dlinks.items()):
                msg.append(dlink.print_card(size))
            for (ID, dresp) in sorted(self.dresps.items()):
                msg.append(dresp.print_card(size))
            for (ID, dvmrel) in sorted(self.dvmrels.items()):
                msg.append(dvmrel.print_card(size))
            for (ID, dvprel) in sorted(self.dvprels.items()):
                msg.append(dvprel.print_card(size))
            for (ID, equation) in sorted(self.dequations.items()):
                msg.append(str(equation))
            if self.doptprm is not None:
                msg.append(self.doptprm.print_card(size))
        return ''.join(msg)

    def _write_tables(self, size):
        """Writes the TABLEx cards sorted by ID"""
        msg = []
        if self.tables:
            msg.append('$TABLES\n')
            for (ID, table) in sorted(self.tables.items()):
                msg.append(table.print_card(size))
        if self.randomTables:
            msg.append('$RANDOM TABLES\n')
            for (ID, table) in sorted(self.randomTables.items()):
                msg.append(table.print_card(size))
        return ''.join(msg)

    def _write_sets(self, size):
        """Writes the SETx cards sorted by ID"""
        msg = []
        if (self.sets or self.setsSuper or self.asets or self.bsets or
            self.csets or self.qsets):
            msg.append('$SETS\n')
            for (ID, setObj) in sorted(self.sets.items()):  # dict
                msg.append(str(setObj))
            for setObj in self.asets:  # list
                msg.append(str(setObj))
            for setObj in self.bsets:  # list
                msg.append(str(setObj))
            for setObj in self.csets:  # list
                msg.append(str(setObj))
            for setObj in self.qsets:  # list
                msg.append(str(setObj))
            for (ID, setObj) in sorted(self.setsSuper.items()):  # dict
                msg.append(str(setObj))
        return ''.join(msg)

    def _write_dynamic(self, size):
        """Writes the dynamic cards sorted by ID"""
        msg = []
        if (self.dareas or self.nlparms or self.frequencies or self.methods or
            self.cMethods or self.tsteps or self.tstepnls):
            msg.append('$DYNAMIC\n')
            for (ID, method) in sorted(self.methods.items()):
                msg.append(method.print_card(size))
            for (ID, cMethod) in sorted(self.cMethods.items()):
                msg.append(cMethod.print_card(size))
            for (ID, darea) in sorted(self.dareas.items()):
                msg.append(darea.print_card(size))
            for (ID, nlparm) in sorted(self.nlparms.items()):
                msg.append(nlparm.print_card(size))
            for (ID, nlpci) in sorted(self.nlpcis.items()):
                msg.append(nlpci.print_card(size))
            for (ID, tstep) in sorted(self.tsteps.items()):
                msg.append(tstep.print_card(size))
            for (ID, tstepnl) in sorted(self.tstepnls.items()):
                msg.append(tstepnl.print_card(size))
            for (ID, freq) in sorted(self.frequencies.items()):
                msg.append(freq.print_card(size))
        return ''.join(msg)

    def _write_aero(self, size):
        """Writes the aero cards"""
        msg = []
        if (self.aero or self.aeros or self.gusts or self.caeros
        or self.paeros or self.trims):
            msg.append('$AERO\n')
            for (ID, caero) in sorted(self.caeros.items()):
                msg.append(caero.print_card(size))
            for (ID, paero) in sorted(self.paeros.items()):
                msg.append(paero.print_card(size))
            for (ID, spline) in sorted(self.splines.items()):
                msg.append(spline.print_card(size))
            for (ID, trim) in sorted(self.trims.items()):
                msg.append(trim.print_card(size))

            for (ID, aero) in sorted(self.aero.items()):
                msg.append(aero.print_card(size))
            for (ID, aero) in sorted(self.aeros.items()):
                msg.append(aero.print_card(size))

            for (ID, gust) in sorted(self.gusts.items()):
                msg.append(gust.print_card(size))
        return ''.join(msg)

    def _write_aero_control(self, size):
        """Writes the aero control surface cards"""
        msg = []
        if (self.aefacts or self.aeparams or self.aelinks or self.aelists or
            self.aestats or self.aesurfs):
            msg.append('$AERO CONTROL SURFACES\n')
            for (ID, aelinks) in sorted(self.aelinks.items()):
                for aelink in aelinks:
                    msg.append(aelink.print_card(size))
            for (ID, aeparam) in sorted(self.aeparams.items()):
                msg.append(aeparam.print_card(size))
            for (ID, aestat) in sorted(self.aestats.items()):
                msg.append(aestat.print_card(size))

            for (ID, aelist) in sorted(self.aelists.items()):
                msg.append(aelist.print_card(size))
            for (ID, aesurf) in sorted(self.aesurfs.items()):
                msg.append(aesurf.print_card(size))
            for (ID, aefact) in sorted(self.aefacts.items()):
                msg.append(aefact.print_card(size))
        return ''.join(msg)

    def _write_flutter(self, size):
        """Writes the flutter cards"""
        msg = []
        if (self.flfacts or self.flutters or self.mkaeros):
            msg.append('$FLUTTER\n')
            for (ID, flfact) in sorted(self.flfacts.items()):
                #if ID!=0:
                msg.append(flfact.print_card(size))
            for (ID, flutter) in sorted(self.flutters.items()):
                msg.append(flutter.print_card(size))
            for mkaero in self.mkaeros:
                msg.append(mkaero.print_card(size))
        return ''.join(msg)

    def _write_thermal(self, size):
        """Writes the thermal cards"""
        msg = []
        # PHBDY
        if self.phbdys or self.convectionProperties or self.bcs:
            # self.thermalProperties or
            msg.append('$THERMAL\n')

            for (key, phbdy) in sorted(self.phbdys.items()):
                msg.append(phbdy.print_card(size))

            #for key,prop in sorted(self.thermalProperties.iteritems()):
            #    msg.append(str(prop))
            for (key, prop) in sorted(self.convectionProperties.items()):
                msg.append(prop.print_card(size))

            # BCs
            for (key, bcs) in sorted(self.bcs.items()):
                for bc in bcs:  # list
                    msg.append(bc.print_card(size))
        return ''.join(msg)

    def _write_coords(self, size):
        """Writes the coordinate cards in a sorted order"""
        msg = []
        if len(self.coords) > 1:
            msg.append('$COORDS\n')
        for (ID, coord) in sorted(self.coords.items()):
            if ID != 0:
                msg.append(coord.print_card(size))
        return ''.join(msg)

    def _write_rejects(self, size):
        """
        Writes the rejected (processed) cards and the rejected unprocessed
        cardLines
        """
        msg = []
        if self.reject_cards:
            msg.append('$REJECTS\n')
            for reject_card in self.reject_cards:
                try:
                    msg.append(print_card(reject_card))
                except RuntimeError:
                    for field in reject_card:
                        if field is not None and '=' in field:
                            raise SyntaxError('cannot reject equal signed '
                                          'cards\ncard=%s\n' % reject_card)
                    raise

        if self.rejects:
            msg.append('$REJECT_LINES\n')
        for reject_lines in self.rejects:
            if reject_lines[0][0] == ' ':
                continue
            else:
                for reject in reject_lines:
                    reject2 = reject.rstrip()
                    if reject2:
                        msg.append(str(reject2) + '\n')
        return ''.join(msg)
