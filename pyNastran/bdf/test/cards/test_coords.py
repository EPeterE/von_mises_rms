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
from numpy import array, allclose
import unittest

from pyNastran.bdf.bdf import BDF


class TestCoords(unittest.TestCase):
    def test_same(self):  # passes
        grids = [
            [0, 0., 0., 1.],
            [0, 0., 1., 0.],
            [0, 1., 0., 0.],
            [0, 1., 1., 1.],
            [0, 1., 1., 0.],
        ]
        grids_expected = grids
        coords = []
        self.getNodes(grids, grids_expected, coords)

    def test_shift(self):  # passes
        grids = [
            [1, 0., 0., 1.],
            [1, 0., 1., 0.],
            [1, 1., 0., 0.],
            [1, 1., 1., 1.],
            [1, 1., 1., 0.],
        ]
        grids_expected = [
            [1, 1., 1., 2.],
            [1, 1., 2., 1.],
            [1, 2., 1., 1.],
            [1, 2., 2., 2.],
            [1, 2., 2., 1.],
        ]

        coords = [  # rid origin,      zaxis,     xaxis
            [0, [1., 1., 1.], [1., 1., 2.], [2., 1., 1.]],
                 ]
        self.getNodes(grids, grids_expected, coords)

    def test_rotate(self):  # passes
        grids = [
                     [1, 0., 0., 1.],
                     [1, 0., 1., 0.],
                     [1, 1., 0., 0.],
                     [1, 1., 1., 1.],
                     [1, 1., 1., 0.],
                 ]
        grids_expected = [
                     #     y    z   x
                     [1., 1., 0, 0.],
                     [1., 0., -1, 0.],
                     [1., 0., 0, 1.],
                     [1., 1., -1, 1.],
                     [1., 0., -1, 1.],
                 ]

        coords = [  # rid origin,      zaxis,     xaxis
                   [0, [0., 0., 0.], [1., 0., 0.], [0., 0., 1.]],
                 ]
        self.getNodes(grids, grids_expected, coords)

    def test_rotate2(self):   # passes
        grids = [
                     [1, 0., 0., 1.],
                     [1, 0., 1., 0.],
                     [1, 1., 0., 0.],
                     [1, 1., 1., 1.],
                     [1, 1., 1., 0.],
                 ]
        grids_expected = [
                     [1, 0., 0., -1.],
                     [1, 0., -1., 0.],
                     [1, 1., 0., 0.],
                     [1, 1., -1., -1.],
                     [1, 1., -1., 0.],
                 ]

        coords = [  # rid origin,     zaxis        xaxis
                   [0, [0., 0., 0.], [0., 0., -1.], [1., 0., 0.]],
                 ]
        self.getNodes(grids, grids_expected, coords)

    def test_rotate3(self):  # passes
        #print('test_rotate3')
        grids = [
                     [1, 0., 0., 1.],
                     [1, 0., 1., 0.],
                     [1, 1., 0., 0.],
                     [1, 1., 1., 1.],
                     [1, 1., 1., 0.],
                 ]
        grids_expected = [
                     [1, 0., 0., -1.],
                     [1, 0., 1., 0.],
                     [1, -1., 0., 0.],
                     [1, -1., 1., -1.],
                     [1, -1., 1., 0.],
                 ]

        coords = [  # rid origin,     zaxis        xaxis
                   [0, [0., 0., 0.], [0., 0., -1.], [-1., 0., 0.]],
                 ]
        self.getNodes(grids, grids_expected, coords)

    @unittest.skip('skipping test_rid_1')
    def test_rid_1(self):  # did i mess up the transform???
        #print('test_rid_1')
        grids = [
                     [2, 10., 5., 3.],  # cid, x,y,z
                    #[3,    10., 5.,  3.],
                 ]
        grids_expected = [
                     ['x', 11., 6., 4.],  # ??? x,y,z
                    #['x',  11., 6.,  4.],
                 ]

        coords = [  # rid origin,     zaxis        xaxis
                   [0, [0., 0., 0.], [0., 0., -1.], [1., 0., 0.]],  # cid=1
                   [1, [1., 1., 1.], [1., 1., 2.], [2., 1., 1.]],  # cid=2
                  #[  1,  [0.,0.,0.], [0.,0., 1.], [1.,0.,0.]  ],  # cid=2,equiv

                   [0, [1., 1., 1.], [1., 1., 2.], [2., 1., 1.]],  # cid=3
                  #[  0,  [0.,0.,0.], [0.,0., 1.], [1.,0.,0.]  ],  # cid=3,equiv
                 ]
        self.getNodes(grids, grids_expected, coords)

    def test_cord2r_1(self):
        grid = ['GRID       20143       7 -9.31-4  .11841 .028296']
        coord = ['CORD2R         7           1.135 .089237  -.0676    .135 .089237  -.0676',
                 '           1.135 .089237   .9324']

        model = BDF()
        card = model.process_card(grid)
        model.add_card(card, card[0])
        card = model.process_card(coord)
        model.add_card(card, card[0])
        model.cross_reference()

        g = model.Node(20143)
        #print(g.Position(debug=False))
        diff = g.Position() - array([1.106704, .207647, -0.068531])

        assert allclose(diff, 0.)

    #def makeNodes(self, grids, coords):
        #grids2 = []

    def getNodes(self, grids, grids_expected, coords, debug=False):
        model = BDF(debug=False)

        for (nid, grid) in enumerate(grids):
            (cid, x, y, z) = grid
            model.add_card(['GRID', nid + 1, cid, x, y, z], 'GRID')
            gridObj = model.Node(nid + 1)
            if debug:
                print(gridObj)

        for (cid, coord) in enumerate(coords):
            #print coord
            (rid, x, y, z) = coord
            model.add_card(['CORD2R', cid + 1, rid] + x + y + z, 'CORD2R')
            coordObj = model.Coord(cid + 1)
            if debug:
                print(coordObj)

        model.cross_reference()

        for (i, grid) in enumerate(grids_expected):
            (cid, x, y, z) = grid
            node = model.Node(i + 1)
            pos = node.Position()
            n = array([x, y, z])

            msg = 'expected=%s actual=%s\n' % (n, pos)
            msg += 'n=%s grid=\n%s' % (i + 1, node)
            coord = node.cp
            msg += 'n=%s coord=\n%s' % (node.nid, coord)
            while coord.rid:
                msg += 'n=%s rcoord=\n%s' % (node.nid, coord.rid)
                coord = coord.rid
            assert allclose(n, pos), msg

if __name__ == '__main__':
    unittest.main()
