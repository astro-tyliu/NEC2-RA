import sys
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from nec2array import (ArrayModel, StructureModel, Deck, Wire, VoltageSource,
                  FreqSteps, ExecutionBlock, RadPatternSpec)

np.set_printoptions(threshold=sys.maxsize)

def test_Deck():
    """\
    Test: create a Deck object which represents a wire excited

    Examples
    --------
    >>> test_Deck()
    True
    """
    d = Deck()
    d.append_card('GW', 0, 7, 0., 0., -.25, 0., 0., .25, 1.0E-5)
    d.append_card('GE', 0)
    d.append_card('EX', 0, 0, 4, 0, 1.0)
    d.append_card('EN')
    d2 = Deck([('GW', 0, 7, 0.0, 0.0, -0.25, 0.0, 0.0, 0.25, 1e-05),
               ('GE', 0),
               ('EX', 0, 0, 4, 0, 1.0),
               ('EN',)])
    print('Decks() are the same?', d==d2)


def test_Deck_load_necfile():
    """
    Test: load in a Deck object from a .nec file

    Examples
    --------
    >>> test_Deck_load_necfile()

    """
    test_necin = \
"""\
CM TESTEX1
CE EXAMPLE 1. CENTER FED LINEAR ANTENNA
CE345678901234567890123456789012345678901234567890123456789012345678901234567890
GW  0    7       0.0        0.      -.25        0.        0.       .25      .001
GE  0
FR  0    2    0    0       50.        2.
EK  1
EX  0    0    4    0        1.        0.        0.        0.        0.        0.
XQ  0
LD  0    0    4    4       10. 3.000E-09 5.300E-11
PQ  0    0    0    0
NE  0    1    1   15      .001        0.        0.        0.        0.    .01786
EN
"""
    d = Deck()
    d.load_necfile(StringIO(test_necin), 'COLUMNS')
    print(d)
    return d


def test_Deck_exec_pynec():
    """\
    Test: create Deck and then execute it using pynec
    """
    d = test_Deck_load_necfile()
    cntxt = next(d.exec_pynec())
    for f in range(2):
        inp_parms = cntxt.get_input_parameters(f)
        print(inp_parms.get_impedance())

def test_StructureModel():
    p1 = (+0.5, 0., 0.1)
    p2 = (+0.5, 0., 0.9)
    p3 = (-0.5, 0., 0.9)
    p4 = (-0.5, 0., 0.1)
    l12 = (p1, p2)
    l23 = (p2, p3)
    l34 = (p3, p4)
    wire_rad = 0.001
    model = StructureModel('Big_Xing')
    model['xing']['X+'] = Wire(*l12, wire_rad)
    model['xing']['mid'] = Wire(*l23, wire_rad)
    model['xing']['X-'] = Wire(*l34, wire_rad)
    model['xing']['mid'].add_port(0.5, 'VS', VoltageSource(1.0))
    fs = FreqSteps('lin', 3, 100., 10.)
    rps = RadPatternSpec()
    ex_ports = [('VS', VoltageSource(1.0))]
    eb = ExecutionBlock(fs, ex_ports, rps)
    model.segmentalize(15, fs.max_freq())
    model.add_executionblock('exblk', eb)
    d = model.as_neccards()
    cntxt = next(d.exec_pynec())
    for f in range(len(fs.aslist())):
        inp_parms = cntxt.get_input_parameters(f)
        for pnr, portname in enumerate(ex_ports):
            print(portname, inp_parms.get_impedance()[pnr])


def test_manual_and_autosegment():
    """\
    Test of manual and autosegmentation
    """
    p1 = (0., 0., +0.5)
    p2 = (0., 0.,  0.0)
    p3 = (0., 0., -0.5)
    l12 = (p1, p2)
    l23 = (p2, p3)
    wire_rad = 0.001
    fs = FreqSteps('lin', 1, 150.)  # MHz
    model = StructureModel('2partDip')
    model['dip']['Z+'] = Wire(*l12, wire_rad)
    model['dip']['Z-'] = Wire(*l23, wire_rad)
    model['dip']['Z-'].add_port(0.0, 'VS')
    eb = ExecutionBlock(fs, [('VS', VoltageSource(1.0))])
    #  Segmentalize manually
    model['dip']['Z+'].nr_seg = 20
    model['dip']['Z-'].nr_seg = 80
    model.add_executionblock('exblk', eb)
    # Converting to neccards assigns port seg nr
    d = model.as_neccards()
    print('Using manual segmentation '
          'Excitation is on seg nr 21:', model['dip']['Z-'].port.ex_seg == 21)
    segsperwavelen = 2*100
    model.segmentalize(segsperwavelen, fs.max_freq())
    modelseglamlens = model.seglamlens()
    # As per previous comment 
    d = model.as_neccards()
    print(f'Using segmentalize() with {segsperwavelen} seg/lambda '
          f'(ie {1/segsperwavelen} lam/seg) '
          f'Achieved lam/seg: {modelseglamlens}\n'
          'Excitation is on seg nr 51:', model['dip']['Z-'].port.ex_seg==51)


def sim_abraham_dip(freq):
    """\
    Create an Abraham dipole for a given frequency
    """
    lamhalf = 1.0
    w_radii = 1e-5*2*lamhalf
    dip_len = lamhalf
    p1 = (0.,0.,-dip_len/2)
    p2 = (0.,0.,+dip_len/2)
    l12 = (p1, p2)
    port_name = 'VS'
    abradip = StructureModel('AbrahamDip')
    abradip['dip']['Z'] = Wire(*l12, w_radii).add_port(0.5, port_name)
    abradip.segmentalize(565, freq)
    return abradip, port_name


def test_EEL():
    """\
    Test: Create an Abraham dipole and simulate its embedded element length

    Ref: Lo 1988 p 6-5.
    """
    fs = FreqSteps('lin', 1, 10.)  # MHz
    abradip, port_name = sim_abraham_dip(fs.max_freq())
    ex_port = (port_name, VoltageSource(1.0))
    rps = RadPatternSpec(nth=1, thets=90., dth=0., nph=1, phis=0., dph=0.)
    eb = ExecutionBlock(fs, [ex_port], rps)
    eepdat = abradip.calc_eep_SC(eb, True)
    eeldat_OC = eepdat.transform_to('OC').get_EELs()
    eeldat= eepdat.get_EELs()
    Hsc_abs = np.sqrt(np.abs(eeldat.eels[0].f_tht)**2
                  +np.abs(eeldat.eels[0].f_phi)**2)
    efflen_theory = abradip['dip']['Z'].length()/2.0
    efflen_simult = Hsc_abs*np.abs(eeldat.eels[0].inp_Z)

    print(eeldat_OC.eels[0].f_tht, np.abs(eeldat.eels[0].inp_Z), Hsc_abs)
    print('Effective length for Abraham dipole...',
          'Simulated:', np.ndarray.item(efflen_simult),
          'Theory:', efflen_theory)
    
def test_SC_OC_transforms():
    fs = FreqSteps('lin', 1, 10.)  # MHz
    abradip, port_name = sim_abraham_dip(fs.max_freq())
    ex_port = (port_name, VoltageSource(1.0))
    rps = RadPatternSpec(nth=1, thets=90., dth=0., nph=1, phis=0., dph=0.)
    eb = ExecutionBlock(fs, [ex_port], rps)
    eep_SC = abradip.calc_eep_SC(eb)
    eep_OC = eep_SC.transform_to('OC')
    eep_OC_SC = eep_OC.transform_to('SC')
    fs2 = FreqSteps('lin', 1, 20.)  # MHz
    eep_SC2 = abradip.calc_eep_SC(ExecutionBlock(fs2, [ex_port], rps))
    print("EEPs. Should be False:", eep_SC2 == eep_SC)
    print("EEPS. Should be True:", eep_OC_SC == eep_SC)
    eel_SC = eep_SC.get_EELs()
    eel_OC_SC = eep_OC_SC.get_EELs()
    eel_SC2 = eep_SC2.get_EELs()
    print("EELs. Should be False:", eel_SC2 == eel_SC)
    print("EELs. Should be True:", eel_OC_SC == eel_SC)
    # Thevenin
    eep_NO = eep_SC.transform_to('NO', adm_load=1/50. * np.identity(1))
    eel_NO = eep_NO.get_EELs()
    print(eel_SC.eels)
    print(eel_NO.eels)


def test_ArrayModel_offcenter():
    lamhalf = 1.0
    w_radii = 1e-5*2*lamhalf
    dip_len = lamhalf
    p1 = (-dip_len/2, 0., 0.)
    p2 = (+dip_len/2, 0., 0.)
    l12 = (p1, p2)
    offcnt = ArrayModel('Off_Center')
    offcnt['dip']['Z'] = Wire(*l12, w_radii).add_port(0.5,'VS')
    fs = FreqSteps('lin', 1, 3e8/(10*2*lamhalf)/1e6)  # MHz
    offcnt.segmentalize(65, fs.max_freq())
    ex_port = ('VS', VoltageSource(1.0))
    rps = RadPatternSpec(nth=3, dth=10., nph=2*2, phis=90.0, dph=45.)
    arr_pos = [[10., 21., 15.]]
    offcnt.arrayify(element=['dip'], array_positions=arr_pos)
    eb = ExecutionBlock(fs, ex_port, rps)
    eepdat = offcnt.excite_1by1(eb, save_necfile=True)
    sv = offcnt.calc_steering_vector(eb)
    ant_nr = 0
    frq_nr = 0
    print('Steering vector phases:')
    ref_ampphs0 = np.conj(sv[ant_nr, frq_nr,0,0])
    print(np.angle(sv[ant_nr, frq_nr]*ref_ampphs0, deg=True))
    ref_ampphs = np.conj(eepdat.eeps[ant_nr].f_phi[frq_nr][0,0])
    _relangs = np.angle(eepdat.eeps[ant_nr].f_phi[frq_nr]*ref_ampphs, deg=True)
    print('Field phase rel. theta=0.')
    print(_relangs)


def lamhalfdip(rad_lam=None):
    lamhalf = 1.0
    if rad_lam is None:
        w_radii = 1e-5*2*lamhalf
    else:
        w_radii = rad_lam * 2*lamhalf
    dip_len = lamhalf
    p1 = (-dip_len/2, 0., 0.5)
    p2 = (+dip_len/2, 0., 0.5)
    l12 = (p1, p2)
    dip = StructureModel('lamhalfdip')
    dip['dip']['X'] = Wire(*l12, w_radii).add_port(0.5,'VS')
    freq = 3e8/(2*lamhalf)
    dip.segmentalize(65, freq/1e6)
    return dip, freq


def test_loaded_lamhalfdip():
    """\
    Test loading of fat dipole over ground (uses extended thin wire kernel)
    """
    dip_fat, freq = lamhalfdip(0.01)
    dip_fat.set_ground()  # Default is a PEC ground in Z=0 plane.
    fs = FreqSteps('lin', 1, freq/1e6)  # MHz
    rps = RadPatternSpec(nth=1, thets=90., nph=0, phis=0.)
    ex_port = ('VS', VoltageSource(1.0))
    # Add "Extended Thin Wire kernel" to execution block
    eb = ExecutionBlock(fs, [ex_port], rps, ext_thinwire=True)
    eepdat_SC = dip_fat.calc_eep_SC(eb)
    print("Diagnostics", dip_fat.seglamlens(), dip_fat.segthinness())
    Y_load = 1.0*1./(78.2-45.7j)
    eepdat_NO = eepdat_SC.transform_to('NO', adm_load=np.atleast_2d(Y_load))
    eepdat_OC = eepdat_SC.transform_to('OC')
    eel_loaded = eepdat_NO.get_EELs()
    eeldat_OC =eepdat_OC.get_EELs()
    print(eepdat_SC.get_impedances())
    print('EEL 50 Ohms', 3e8/freq/np.pi, np.abs(eeldat_OC.eels[0].f_tht).item())
    print('EEL 50 Ohms', np.abs(eel_loaded.eels[0].f_tht).item())
    print(377*Y_load*(np.abs(eel_loaded.eels[0].f_tht).item())**2)


def two_lamhalfdip():
    lamhalf = 1.0
    w_radii = 1e-5*2*lamhalf
    dip_len = lamhalf
    p1 = (0., 0., -dip_len/2)
    p2 = (0., 0., +dip_len/2)
    l12 = (p1, p2)
    twodip = ArrayModel('2dip_sbs')
    twodip['dip']['Z'] = Wire(*l12, w_radii).add_port(0.5,'VS')
    return twodip


def test_Array_2_lamhalfdip_sbys():
    """
    Test of two lambda/2 dipole side-by-side array

    See Balanis Antenna Theory 2016, Fig 8.21
    """
    # Use function to build model of lambda half dipole
    twodip = two_lamhalfdip()
    # Get the port name...
    _dipgrpname = list(twodip.groups.keys()).pop()
    portname = list(twodip[_dipgrpname].get_ports().keys()).pop()
    # ...and the length of the dipole
    lamhalf = twodip[_dipgrpname].total_length()
    fs = FreqSteps('lin', 1, 3e8/(2*lamhalf)/1e6)  # MHz
    twodip.segmentalize(65, fs.max_freq())
    ex_port = (portname, VoltageSource(1.0))
    dists = 2*lamhalf * np.linspace(0.0, 3., 50)[1:]
    mutimp = []
    for dist in dists:
        arr_pos = [[0.,0.,0.], [dist, 0., 0.]]
        twodip.arrayify(element=['dip'],
                        array_positions=arr_pos)
        eepdat = twodip.excite_1by1(ExecutionBlock(fs, ex_port))
        impmat = eepdat.get_impedances()
        mutimp.append(impmat[0,0,1])
    mutimp = np.array(mutimp)
    # Get wire radius
    _partname = list(twodip[_dipgrpname].parts.keys()).pop()
    w_radii = twodip[_dipgrpname][_partname].radius
    plt.plot(dists/(2*lamhalf), np.real(mutimp))
    plt.plot(dists/(2*lamhalf), np.imag(mutimp))
    plt.legend(['Resistive','Reactive'])
    plt.xlabel('Separation [lambda]')
    plt.ylabel('Mutual-impedance [Ohm]')
    plt.title('2 side-by-side half-wave dipoles\n'
              'simulated with NEC2 '
              f'(wire radius={w_radii/(2*lamhalf)} lambda)')
    plt.show()

def test_get_antspats():
    """
    Test of getting the antenna patterns tensor from an EEPdat object
    """
    # Use function to build model of lambda half dipole
    twodip = two_lamhalfdip()
    # Get the port name...
    _dipgrpname = list(twodip.groups.keys()).pop()
    portname = list(twodip[_dipgrpname].get_ports().keys()).pop()
    # ...and the length of the dipole
    lamhalf = twodip[_dipgrpname].total_length()
    fs = FreqSteps('lin', 1, 3e8/(2*lamhalf)/1e6)  # MHz
    nph=36
    rps = RadPatternSpec(nth=1, thets=90., nph=nph, phis=0., dph=360/nph)
    twodip.segmentalize(65, fs.max_freq())
    ex_port = (portname, VoltageSource(1.0))
    arr_pos = [[0.,0.,0.], [lamhalf,0.,0.]]
    twodip.arrayify(element=['dip'],
                        array_positions=arr_pos)
    eepdat = twodip.excite_1by1(ExecutionBlock(fs, ex_port, rps))
    antspats_arr = eepdat.get_antspats_arr()
    ant_nr = 0
    frq_nr = 0
    theta_nr = 0
    polcomps = {'theta':0, 'phi':1}
    polcomp = 'theta'
    plt.plot(np.abs(antspats_arr[ant_nr,frq_nr,theta_nr,:,polcomps[polcomp]]))
    plt.title('Rad pattern, XY plane cut, Two dipoles')
    plt.legend([f'abs(E_{polcomp})'])
    plt.show()


test_Deck()
test_Deck_load_necfile()
test_Deck_exec_pynec()
test_StructureModel()
test_manual_and_autosegment()
test_EEL()
test_SC_OC_transforms()
test_ArrayModel_offcenter()
test_loaded_lamhalfdip()
test_Array_2_lamhalfdip_sbys()
test_get_antspats()

