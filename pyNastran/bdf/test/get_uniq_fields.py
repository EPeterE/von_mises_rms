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
#!/usr/bin/python
import sys
from pyNastran.bdf.bdf import BDF, to_fields, BDFCard, interpret_value, wipe_empty_fields
import logging

logger = logging.getLogger("bdfuniq")
logger.setLevel(logging.DEBUG)

log_formater = logging.Formatter(
    fmt="%(name)-10s: %(levelname)-5s: %(message)s")

log_handler_stderr = logging.StreamHandler()
log_handler_stderr.setFormatter(log_formater)
logger.addHandler(log_handler_stderr)

log_handler_file = logging.FileHandler("bdf_get_uniq.log", "a")
log_handler_file.setFormatter(log_formater)
logger.addHandler(log_handler_file)

includeDir = None


class BDFuniqCard(BDF):
    def __init__(self, log, fingset):
        self.card_set = fingset
        self.f = open('cards.out.bdf', 'w')
        #self.newBDF = newBDF
        BDF.__init__(self, log=log)

    #def _parse_executive_control_deck(self):
    #   pass

    def cross_reference(self, xref):
        pass

    def add_card(self, card_lines, card_name, comment=''):
        card = to_fields(card_lines, card_name)
        card = [interpret_value(val) for val in card]
        card = BDFCard(card)
        
        #if cardName == "LOAD":
        rec = []
        for item in card:
            item_type = type(item)
            if item_type == int:
                item = 1
                #print "int"
            elif item_type == float:
                item = 1.0
                #print "float"
            rec.append(item)
            #print type(c), c
        #print "*", rec
        rec_str = str(rec)
        if not rec_str in self.card_set:
            #self.newBDF.add_card(card, cardName, iCard, old_card_obj)
            #print(card)
            self.card_set.add(rec_str)

    def is_reject(self, cardName):
        """Can the card be read"""
        #cardName = self._get_card_name(card)
        if cardName.startswith('='):
            return False
        elif not cardName in self.cardsToRead:
            if cardName:
                if cardName not in self.reject_count:
                    self.log.info("RejectCardName = |%s|" % cardName)
                    self.reject_count[cardName] = 0
                self.reject_count[cardName] += 1
        return False

    def write_bdf(self):
        for card in self.card_set:
            self.f.write(str(card) + '\n')
        self.f.close()
        

if __name__ == '__main__':
    print("enter list of filenames at the command line...")
    #newBDF = BDF()
    card_fingerprint_set = set()
    infilenames = sys.argv[1:]
    assert len(infilenames) > 0, infilenames
    for infilename in infilenames:
        print("running infilename=%r" % infilename)
        try:
            infilename = infilename.strip()
            sys.stdout.flush()
            model = BDFuniqCard(log=logger,
                        fingset=card_fingerprint_set)
            model.read_bdf(infilename)
        except BaseException as e:
            logger.error(e)
    print("please see cards.out.bdf")
    model.write_bdf()
    print("done...")
