"""Avoid route-budget work for contacts outside every common elevation band."""
import unittest
from terminal_junction_consensus import choose_terminal_band


class TerminalBandPrefilterTests(unittest.TestCase):
    def test_impossible_bands_do_not_spend_route_validation_budget(self):
        def point(z):return dict(z=float(z),terminal_retreat_m=0.,xyz_distance_m=.001,u=0.)
        calls=[]
        def valid(i,p):calls.append((i,p['z']));return True
        valid.breakpoints=lambda i,p:[]
        chosen=choose_terminal_band([[point(z) for z in range(100)],[point(50)]],valid=valid)
        self.assertEqual([p['z'] for p in chosen],[50.,50.])
        self.assertLess(len(calls),10)

    def test_disconnected_feasible_bands_keep_terminal_preference(self):
        def point(z,r):return dict(z=float(z),terminal_retreat_m=r,xyz_distance_m=.001,u=0.)
        chosen=choose_terminal_band([[point(3,2),point(9,0)],[point(3.05,1),point(9.05,0)]])
        self.assertEqual([p['z'] for p in chosen],[9.,9.05])


if __name__=='__main__':unittest.main()
