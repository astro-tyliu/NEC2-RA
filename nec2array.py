from copy import deepcopy
from dataclasses import dataclass, astuple
import typing
import pathlib
import numpy as np
import warnings

C0 = 2.99792458e8  # Speed of light
MU0 = 4*np.pi*1e-7  # H/m aka vacuum magnetic permeability
ETA0 = MU0 * C0  # Impedance of free space

PRGINPS = {'STGEOM', 'PCNTRL'}
CARDS_COMMT = {'CE', 'CM'}
CARDS_GEOMT = {'GA', 'GE', 'GF', 'GH', 'GM', 'GR', 'GS', 'GW', 'GX',
               'SP', 'SM'}
CARDS_CNTRL = {'CP', 'EK', 'EN', 'EX', 'FR', 'GD', 'GN', 'KH', 'LD',
               'NE', 'NH', 'NT', 'NX', 'PQ', 'PT', 'RP', 'TL', 'WG', 'XQ'}
CNTRL_COLS = [2,    5,    10,   15,   20,   30,   40,   50,   60, 70]
GEOMT_COLS = [2,    5,    10,   20,   30,   40,   50,   60,   70, 80]
PARLBLS = {'STGEOM': ['I1', 'I2', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7'],
           'PCNTRL': ['I1', 'I2', 'I3', 'I4', 'F1', 'F2', 'F3', 'F4', 'F5',
                      'F6']}
CARDDEFS = {
    'CM': {'PRGINP': 'COMMNT'},
    'CE': {'PRGINP': 'COMMNT'},

    'GA': {'PRGINP': 'STGEOM', 
           'I1': 'ITG', 'I2': 'NS',
           'F1': 'RADA', 'F2': 'ANG1', 'F3': 'ANG2', 'F4': 'RAD'},
    'GE': {'PRGINP': 'STGEOM',
           'I1': 'GPFLAG'},
    'GH': {'PRGINP': 'STGEOM',
           'I1': 'ITG', 'I2': 'NS', 'F1': 'S', 'F2': 'HL', 'F3': 'A1',
           'F4': 'B1', 'F5': 'A2', 'F6': 'B2', 'F7': 'RAD'},
    'GM': {'PRGINP': 'STGEOM',
           'I1': 'ITGI', 'I2': 'NRPT', 'F1': 'ROX', 'F2': 'ROY', 'F3': 'ROZ',
           'F4': 'XS', 'F5': 'YS', 'F6': 'ZS', 'F7': 'ITS'},
    'GW': {'PRGINP': 'STGEOM',
           'I1': 'ITG', 'I2':  'NS', 'F1': 'XW1', 'F2': 'YW1',
           'F3': 'ZW1', 'F4': 'XW2', 'F5': 'YW2', 'F6': 'ZW2',
           'F7': 'RAD'},

    'CP': {'PRGINP': 'PCNTRL',
           'I1': 'TAG1', 'I2': 'SEG1', 'I3': 'TAG2', 'I4': 'SEG2'},
    'EK': {'PRGINP': 'PCNTRL',
           'I1': 'ITMP1'},
    'EN': {'PRGINP': 'PCNTRL'},
    'EX': {'PRGINP': 'PCNTRL',
           'I1': 'I1', 'I2': 'I2', 'I3': 'I3', 'I4': 'I4',
           'F1': 'F1', 'F2': 'F2', 'F3': 'F3',
           'F4': 'F4', 'F5': 'F5', 'F6': 'F6'},       
    'FR': {'PRGINP': 'PCNTRL',
           'I1': 'IFRQ', 'I2': 'NFRQ', 'F1': 'FMHZ', 'F2': 'DELFRQ'},
    'GD': {'PRGINP': 'PCNTRL',
           'F1': 'EPSR2', 'F2': 'SIG2', 'F3': 'CLT', 'F4': 'CHT'},
    'GN': {'PRGINP': 'PCNTRL',
           'I1': 'IPERF', 'I2': 'NRADL', 'I3': 'BLANK', 'I4': 'BLANK',
           'F1': 'EPSE', 'F2': 'SIG', 'F3': 'F3', 'F4': 'F4', 'F5': 'F5',
           'F6': 'F6'},
    'KH': {'PRGINP': 'PCNTRL',
           'F1': 'RKH'},
    'LD': {'PRGINP': 'PCNTRL',
           'I1': 'LDTYP', 'I2': 'LDTAG', 'I3': 'LDTAGF', 'I4': 'LDTAGT',
           'F1': 'ZLR', 'F2': 'ZLI', 'F3': 'ZLC'},
    'NE': {'PRGINP': 'PCNTRL',
           'I1': 'NEAR', 'I2': 'NRX', 'I3': 'NRY', 'I4': 'NRZ',
           'F1': 'XNR', 'F2': 'YNR', 'F3': 'ZNR',
           'F4': 'DXNR', 'F5': 'DYNR', 'F6': 'DZNR'},
    'NH': {'PRGINP': 'PCNTRL',
           'I1': 'NEAR', 'I2': 'NRX', 'I3': 'NRY', 'I4': 'NRZ',
           'F1': 'XNR', 'F2': 'YNR', 'F3': 'ZNR',
           'F4': 'DXNR', 'F5': 'DYNR', 'F6': 'DZNR'},
    'RP': {'PRGINP': 'PCNTRL',
           'I1': 'I1', 'I2': 'NTH', 'I3': 'NPH', 'I4': 'XNDA',
           'F1': 'THETS', 'F2': 'PHIS', 'F3': 'DTH', 'F4': 'DPH',
           'F5': 'RFLD', 'F6': 'GNOR'},
    'PQ': {'PRGINP': 'PCNTRL',
           'I1': 'IPTFLQ', 'I2': 'IPTAQ', 'I3': 'IPTAQF', 'I4': 'IPTAQT'},
    'XQ': {'PRGINP': 'PCNTRL',
           'I1': 'I1'}
}


class Deck:
    """
    NEC deck of cards
    """
    def __init__(self, cardlist=None) -> None:
        self.carddeck = []
        if cardlist is not None:
            self.carddeck = cardlist

    def append_card(self, *cardargs):
        self.carddeck.append(cardargs)
    
    def append_cards(self, cardslist):
        self.carddeck.extend(cardslist)
    
    def get_sections(self):
        commt_cards = [_c for _c in self.carddeck if _c[0] in CARDS_COMMT]
        geomt_cards = [_c for _c in self.carddeck if _c[0] in CARDS_GEOMT]
        cntrl_cards = [_c for _c in self.carddeck if _c[0] in CARDS_CNTRL]
        return commt_cards, geomt_cards, cntrl_cards
    
    @classmethod
    def load_necfile_cls(cls, file, cardformat='COLUMNS'):
        deck =  cls()
        for cardstr in file.readlines():
            cardargs = cls._cardstr2args(cardstr, cardformat)
            deck.append_card(*cardargs)

    def load_necfile(self, file, cardformat='COLUMNS'):
        self.__init__()
        for cardstr in file.readlines():
            cardargs = self.__class__._cardstr2args(cardstr, cardformat)
            self.append_card(*cardargs)
        return self
    
    def save_necfile(self, file):
        necfile_suffix = '.nec'
        if type(file) == str:
            _suff = pathlib.Path(file).suffix
            if _suff:
                if _suff != necfile_suffix:
                    file = file.replace(_suff, necfile_suffix)
            else:
                file = file + necfile_suffix
            f = open(file, 'w')
        else:
            f = file
        f.write(str(self))
        f.close()
    
    def as_pynec(self):
        pynec_code_list = []
        _code_section = []
        # Set up PyNEC module and context and geometry
        _code_section.append("import PyNEC")
        _code_section.append("_nec_context = PyNEC.nec_context()")
        _code_section.append("_nec_geom = _nec_context.get_geometry()")
        for card in self.carddeck:
            mn_id = card[0]
            parms = card[1:]
            if mn_id in CARDS_COMMT: continue
            if mn_id in CARDS_GEOMT:
                _arg = parms + (1.0, 1.0)
                if mn_id == 'GM':
                    parms = (parms[2], parms[3], parms[4],
                             parms[5], parms[6], parms[7],
                             parms[8], parms[1], parms[0])
                    _code_section.append(f"_nec_geom.move{parms}")
                if mn_id == 'GW':
                    _code_section.append(f"_nec_geom.wire{_arg}")
                if mn_id == 'GE':
                    _code_section.append(
                        f"_nec_context.geometry_complete{parms}")
            if mn_id in CARDS_CNTRL:
                if mn_id == 'EK':
                    _code_section.append(
                        "_nec_context.set_extended_thin_wire_kernel(True)")
                if mn_id == 'EX':
                    itmp3, itmp4 = self._split_digits(parms[3], 2)
                    _arg = (parms[:3]+(itmp3, itmp4)+parms[4:])
                    _code_section.append(f"_nec_context.ex_card{_arg}")
                if mn_id == 'EN':
                    pass
                if mn_id == 'FR':
                    _code_section.append(f"_nec_context.fr_card{parms}")
                if mn_id == 'GN':
                    parms = parms[:2] + parms[4:]  # Cols I3,I4 not used PyNEC
                    _code_section.append(f"_nec_context.gn_card{parms}")
                if mn_id == 'RP':
                    X, N, D, A = self._split_digits(parms[3], 4)
                    _arg = parms[:3]+(X, N, D, A)+parms[4:]
                    _code_section.append(f"_nec_context.rp_card{_arg}")
                    pynec_code_list.append(_code_section)
                    _code_section = []
                if mn_id == 'XQ':
                    _code_section.append("_nec_context.xq_card(0)")
                    pynec_code_list.append(_code_section)
                    _code_section = []
        if len(pynec_code_list) == 0:
            pynec_code_list.append(_code_section)
        # Chop up cards into sections that end with executions
        pynec_code_strs = []
        for _code_section in pynec_code_list:
            pynec_code_strs.append("\n".join(_code_section))

        return pynec_code_strs
    
    def exec_pynec(self, printout=False):
        _locals = {}
        _code = self.as_pynec()
        _nrblcks = len(_code)
        _blcknr = 0
        for _blck in _code:
            _blcknr += 1
            if printout:
                print(f'**** Executing code block {_blcknr}/{_nrblcks}:')
                print(_blck)
            exec(_blck, None, _locals)
            if printout:
                print('**** Finished code block.')
            _nec_context_ = _locals['_nec_context']
            yield _nec_context_
        exec('del _nec_context', None, _locals)

    @classmethod
    def _cardstr2args(cls, cardstr, cardformat='COLUMNS'):

        def split_cols(l, pnames, lbls):
            cardparms = []
            for lbl in lbls:
                nrcols, ntype = cls._parmcolwidth(lbl)
                colstr = l[:nrcols]
                parmtoadd = None
                if lbl in pnames and colstr:
                    try:
                        parmtoadd = ntype(colstr)
                    except ValueError:
                        raise ValueError(
                                f'Card {mn_id}, param {lbl} is corrupt')
                if parmtoadd is not None:
                    cardparms.append(parmtoadd)
                l = l[nrcols:]
            return tuple(cardparms)
        
        def split_csv(l, pnames, lbls):
            cardparms = []
            pnames = list(pnames)
            _parms = l.split()
            lbl2parm = dict(zip(lbls, _parms))
            for lbl in lbl2parm:
                if lbl in pnames:
                    _, ntype = cls._parmcolwidth(lbl)
                    cardparms.append(ntype(lbl2parm[lbl]))
            return tuple(cardparms)
        
        l = cardstr.rstrip()
        card = []
        mn_id = l[:2]
        l = l[2:]
        if mn_id not in CARDDEFS.keys():
            raise ValueError(f"Memnonic id {mn_id} not valid")
        card.append(mn_id)
        if mn_id == 'CM' or mn_id == 'CE':
            card.append(l)
            return tuple(card)
        lbls = PARLBLS[CARDDEFS[mn_id]['PRGINP']]
        pnames = list(filter(lambda _pn : _pn in lbls, CARDDEFS[mn_id]))
        if cardformat == 'COLUMNS':
            card = (mn_id, *split_cols(l, pnames, lbls))
        else:
            card = (mn_id, *split_csv(l, pnames, lbls))
        return card
    
    def _split_digits(self, int_int, nrdigits):
        int_str = f"{int_int :0{nrdigits}d}"
        intcharlist = [*int_str]
        int_list = [int(d) for d in intcharlist]
        return tuple(int_list)
    
    @staticmethod
    def _parmcolwidth(parmlbl):
        numtyp = parmlbl[0]
        if numtyp == 'I':
            ntype = int
            nrcols = 5
            if parmlbl[1] == '1':
                nrcols = 3
        elif numtyp == 'F':
            ntype = float
            nrcols = 10
        return nrcols, ntype
    
    def __iter__(self):
        for card in self.carddeck:
            yield card
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__): return False
        return self.carddeck == other.carddeck
    
    def __ne__(self, other):
        if not isinstance(other, self.__class__): return True
        return self.carddeck == other.carddeck

    def __str__(self):
        lines = []
        #lines += [f'CM *** Created using {__name__}.{__class__.__name__} ***']
        for card in self.carddeck:
            card = list(card)
            mn_id = card.pop(0)
            if mn_id == 'CM' or mn_id == 'CE':
                line = mn_id + card.pop(0)
                lines.append(line)
                continue
            lbls = PARLBLS[CARDDEFS[mn_id]['PRGINP']]
            pnames = list(filter(lambda _pn : _pn in lbls, CARDDEFS[mn_id]))
            line = mn_id  # Initialize output line
            for lbl in lbls:
                nrcols, ntype = __class__._parmcolwidth(lbl)
                nfmt = 'd' if ntype is int else 'g'
                if lbl in pnames:
                    parm = ntype(card.pop(0))
                else:
                    parm = ntype(0)
                parmstr = f"{parm:>{nrcols}{nfmt}}"
                line += parmstr
            lines.append(line)
        _str = '\n'.join(lines) + '\n'
        return _str
    
    def __repr__(self) -> str:
        repr_ = self.__class__.__name__+'('+ repr(self.carddeck)+")"
        return repr_


@dataclass
class VoltageSource:
    value: complex = 1+0j
    type: str = 'applied-E-field'  # or 'current-slope-discontinuity'

    def nec_type(self):
        ex_i1 = 0 if self.type == 'applied-E-field' else 1
        return ex_i1


@dataclass
class Port:
    fractional_position: float = 0.5
    name: str = '' # new
    source: VoltageSource = VoltageSource()


@dataclass
class Wire:
    point_src: typing.Tuple
    point_dst: typing.Tuple
    radius: float
    nr_seg: int = 0
    port: Port = None

    def parametric_seg(self, fractional_position):
        seg_nr = int(fractional_position * self.nr_seg)+1
        return seg_nr
    
    def add_port(self, fraction_position, port_name='', source=None):
        self.port = Port(fraction_position, port_name, source)
        return self
    
    def length(self):
        _length = np.linalg.norm(np.array(self.point_dst)
                                 - np.array(self.point_src))
        return _length


@dataclass
class Transformations:
    rox: float
    roy: float
    roz: float
    xs: float
    ys: float
    zs: float
    nrpt: int = 0
    itgi: int = 0
    its: int = 0


@dataclass
class FreqSteps:
    steptype: str = 'lin'  # linear or 'exp' multiplicative
    nrsteps: int = 1
    start: float = 299.8  # MHz
    incr: float = 1.

    def aslist(self, MHz=True):
        frqlist = []
        for fi in range(self.nrsteps):
            if self.steptype == 'lin':
                frq = self.start+self.incr*fi
            else:
                frq = self.start*self.incr**fi
            frqlist.append(frq)
        if not MHz:
            frqlist = [_f*1e6 for _f in frqlist]
        return frqlist

    def max_freq(self):
        _max_freq = (self.start+self.incr*(self.nrsteps-1)
                     if self.steptype == 'lin'
                     else self.start*self.incr**(self.nrsteps-1))
        return _max_freq
    
    def to_nec_type(self):
        if self.steptype == 'lin':
            return 0
        elif self.steptype == 'exp':
            return 1
    
    def from_nec_type(self):
        pass


@dataclass
class RadPatternSpec:
    calcmode: int = 0   # 0: normal-mode, 1: surface-wave, 2: linear-cliff
                        # 3: circular-cliff, 4: radial-ground-screen
                        # 5: rad-grd-scr-lin-clff, 6: rad-grd-scr-crc-clff
    nth: int = 1
    nph: int = 1
    xnda: int = 1000    # f"{X}{N}{D}{A}" where:
                        # X=0 major & minor axis, =1 vertical, horizontal
                        # N=0 no normalized gain
                        # D=
                        # A=0 no averaging, 
    thets: float = 0.
    phis: float = 0.
    dth: float = 0.
    dph: float = 0.
    rfdl: float = 0.    # Radial distance, =0 factor exp(-jkr)/r dropped
    gnor: float = 0.    # Gain normalization factor see N

    def hemisphere(self, nth=40, nph=10):
        self.nth = nth
        self.nph = nph
        self.dth = int(90/nth)
        self.dph = int(360/nph)
        return self
    
    def as_thetaphis(self):
        nth = self.nth if self.nth > 0 else 1
        nph = self.nph if self.nph > 0 else 1
        thetas = [self.thets+thtnr*self.dth for thtnr in range(nth)]
        phis = [self.phis+phinr*self.dph for phinr in range(nph)]
        return thetas, phis
    
    def as_thetaphimeshs(self):
        thetas, phis = self.as_thetaphis()
        thetamesh, phimesh = np.meshgrid(thetas, phis, indexing='ij')
        return thetamesh, phimesh
    
    def as_khat(self):
        thetamsh, phimsh = self.as_thetaphimeshs()
        thetamsh = np.deg2rad(thetamsh)
        phimsh = np.deg2rad(phimsh)
        khat = np.array([np.sin(thetamsh)*np.cos(phimsh),
                         np.sin(thetamsh)*np.sin(phimsh),
                         np.cos(thetamsh)])
        khat = np.moveaxis(khat, 0,-1)
        return khat


@dataclass
class Ground:
    iperf: int
    nradl: int
    epse: float
    sig: float
    f3: float = 0.
    f4: float = 0.
    f5: float = 0.
    f6: float = 0.

    def astuple(self):
        return (self.iperf, self.nradl, 0, 0, self.epse, self.sig,
                self.f3, self.f4, self.f5, self.f6)


@dataclass
class ExecutionBlock:
    freqsteps: FreqSteps = FreqSteps()  # Should at least have this
    exciteports: typing.List = None
    radpat: RadPatternSpec = None
    ext_thinwire: bool = False

    def nrexcitedports(self):
        return len(self.exciteports)


@dataclass
class NECout:
    freqs: typing.List
    thetas: typing.List
    phis: typing.List
    f_tht: typing.List
    f_phi: typing.List
    f_type: str = 'Electric'
    inp_V: complex = None
    inp_I: complex = None
    inp_Z: complex = None


class StructureCurrents:
    def __init__(self, freqs, nr_ants):
        self.freqs = freqs
        self.impedance = np.zeros((nr_ants, nr_ants))
        self.currents = []
        self._segtags = []
        self._segnums = []

    def set_currents(self, currents):
        self.currents = currents

    def set_segtags(self, segtags):
        self._segtags = segtags

    def set_segnums(self, segnums):
        self._segnums = segnums

    def _current_index(self, tag, seg):
        # Get the absolute segment nums with excited tag
        abstagsegs = self._segnums[
            np.nonzero(self._segtags == tag)[0]]
        absseg = abstagsegs[0]+seg-1
        return absseg-1
    
    def get_current(self, tag, seg):
        return self.currents[self._current_index(tag, seg)]


def impedanceRLC(freqs, R, L, C, coupling, imp_not_adm=True):
    """
    Compute impedance of a R-L-C circuit
    
    Parameters
    ----------
    freqs: array
        Frequencies in Hz.
    R: float
        Resistance in Ohms. If not used set to `None`.
    L: float
        Inductance in Henries. If not used set to `None`.
    C: float
        Capacitance in Farads. If not used set to `None`.
    coupling: str
        Circuit coupling of the lumped circuit. Can be 'parallel' or 'series'.
    imp_not_adm: bool
        Whether to output responce as a impedance (default), when True,
        or admittance, when False.
    
    Returns
    -------
    imp | adm: array
        Impedance or admittance as of function of the input frequencies.
    """
    imp = None
    adm = None
    omegas = 2*np.pi*np.asarray(freqs)
    if coupling=='parallel':
        Y_R = 0.
        Y_L = 0.
        Y_C = 0.
        if R is not None:
            Y_R = 1/R * np.ones_like(omegas)
        if L is not None:
            Y_L = 1/(1j*omegas*L)
        if C is not None:
            Y_C = 1j*omegas*C
        adm = Y_R + Y_L + Y_C
    elif coupling=='series':
        Z_R = 0.
        Z_L = 0.
        Z_C = 0.
        if R is not None:
            Z_R = R * np.ones_like(omegas)
        if L is not None:
            Z_L = 1j*omegas*L
        if C is not None:
            Z_C = 1/(1j*omegas*C)
        imp = Z_R + Z_L + Z_C
    else:
        raise ValueError(f"coupling='{coupling}' not valid."
                          "(try 'parallel' or 'series')")
    if L and C:
        print("Resonance frequency is:", 1/np.sqrt(L*C)/(2*np.pi),'Hz')
    if imp is None and imp_not_adm:
        imp = 1/adm
    if adm is None and not imp_not_adm:
        adm = 1/imp
    if imp_not_adm:
        return imp
    else:
        return adm


class EEPdata:
    """\
    Class for Embedded Element Pattern data

    Superclass to EEP_SC, EEP_OC, EEP_NO, EEP_TH and EEL
    """
    def __init__(self, eeps, adm_or_imp, excite_typ='SC', adm_or_imp_load=None,
                 excite_val=1.0):
        self.eeps = eeps  # One NecOut (ie EEP) for each excitation (ie element)
        # 'OC' open-circuit implies current
        # 'SC' short-circuit implies voltage
        self.excite_typ = excite_typ
        self.excite_val = excite_val  # Amplitude
        self.adm_or_imp = adm_or_imp  # admittance if SC, impedance if OC
        self.adm_or_imp_load = adm_or_imp_load

    def get_admittances(self):
        """\
        Return the array's admittances

        Returns
        -------
        Admittances has shape (nfrq, nths, nphs, nant, nant)
        """
        if self.excite_typ == 'SC' or self.excite_typ == 'NO':
            adm = self.adm_or_imp
            return adm
        elif self.excite_typ == 'OC' or self.excite_typ == 'TH':
            imp = self.adm_or_imp
            return np.linalg.inv(imp)  
    
    def get_impedances(self):
        if self.excite_typ == 'OC' or self.excite_typ == 'TH':
            imp = self.adm_or_imp
            return imp
        elif self.excite_typ == 'SC' or self.excite_typ == 'NO':
            adm = self.adm_or_imp
            return np.linalg.inv(adm)

    def _get_embedded_elements(self):
        return self.eeps
    
    def get_antspats_arr(self):
        """\
        Get EEPs as one big array

        Returns
        -------
        antspats: array
            The radiation patterns for all antennas.
            The indices are [antnr, freqnr, thetanr, phinr, polnr]
            polnr=0 is theta and polnr=1 is phi.
        """
        eep_list = self._get_embedded_elements()
        f_tht_mat = np.array([np.atleast_3d(_nec.f_tht) for _nec in eep_list])
        f_phi_mat = np.array([np.atleast_3d(_nec.f_phi) for _nec in eep_list])
        antspats = np.stack((f_tht_mat, f_phi_mat), axis=-1)
        if antspats.shape[-3] == 0:
            # In the case that there is no radiation patterns,
            # the atleast_3d() above makes the -2 axis shape 1,
            # but here I force it to 0 to clearify that there is no theta, phis
            antspats = antspats.reshape(
                antspats.shape[:-3] + (0, 0, antspats.shape[-1]))
        return antspats
    
    def set_antspat_arr(self, antspat):
        for antnr, _eep in enumerate(self._get_embedded_elements()):
            _eep.f_tht = antspat[antnr, ..., 0]
            _eep.f_phi = antspat[antnr, ..., 1]

    def get_pow_arr(self):
        if self.excite_typ == 'SC':
            _volt_exc = self.voltage_excite
            _adm = self.admittances
            # Warnick2021 eq. 14
            pow_mat = np.abs(_volt_exc)**2*np.real(_adm)/2.
        elif self.excite_typ == 'OC':
            _curr_exc = self.current_excite
            _imp = self.impedances
            # Warncik2021 eq. 13 (with no loss)
            pow_mat = np.abs(_curr_exc)**2*np.real(_imp)/2.
        else:
            raise NotImplementedError(
                'Array power not implemented for cicuit type {}'
                .format(self.excite_typ))
        return pow_mat

    def get_EELs(self):
        """\
        Calculate Embedded Element Lengths from EEPs
        """
        adm_or_imp_load = None
        if self.excite_typ == 'SC':
            excite_val = self.voltage_excite
            adm_or_imp = self.admittances
        elif self.excite_typ == 'OC':
            excite_val = self.current_excite
            adm_or_imp = self.impedances
        elif self.excite_typ == 'NO':
            excite_val = self.current_excite
            adm_or_imp = self.admittances
            adm_or_imp_load = self.adm_load
        elif self.excite_typ == 'TH':
            excite_val = self.voltage_excite
            adm_or_imp = self.impedances
            adm_or_imp_load = self.imp_load
        freqs = self.eeps[0].freqs
        freqs = np.array(freqs)[:, np.newaxis, np.newaxis, np.newaxis]  # Brdcst
        # Template EELdata initialized with self.eeps which get overwritten
        # after tranformated eeps are calculated.
        # Also need to copy epps and adm_or_imp so return obj can be modified
        # independently of self.
        _ees = deepcopy(self.eeps)
        for _ee in _ees:
            if self.excite_typ == 'OC' or self.excite_typ == 'NO':
                _ee.f_type = 'Length'
            elif self.excite_typ == 'SC' or self.excite_typ == 'TH':
                _ee.f_type = 'Length/Impedance'
        eeldata = EELdata(_ees, np.copy(adm_or_imp),
                          self.excite_typ, np.copy(adm_or_imp_load))
        antspats = self.get_antspats_arr()
        antspats = 2.j/(MU0*freqs*1e6*excite_val)*antspats  # 1e6 = MHz to Hz
        if self.excite_typ == 'TH':
            antspats = np.expand_dims(np.moveaxis(antspats, [0], [-1]), -1)
            antspats = adm_or_imp_load @ antspats
            antspats = np.moveaxis(antspats.squeeze(-1), [-1], [0])
        eeldata.set_antspat_arr(antspats)
        return eeldata
    
    def rectifying_phase(self):
        """\
        Compute the rectifying phase

        Calculates the rectifying phase of the complex vector field F
        using phs=arg(F.F)/2.

        Returns
        -------
        phs_rct : ndarray
            Rectifying phase field. The component-shape is [ant, frq, tht, phi].

        antspats_rct : ndarray
            Rectified complex vector pattern field.
            The component-shape is [ant, frq, tht, phi, pol].
        """
        antspats = self.get_antspats_arr()
        phs_rct = np.angle(antspats[..., 0]**2 + antspats[..., 1]**2)/2
        antspat_rct = antspats * np.exp(-1j*phs_rct[..., np.newaxis])
        return phs_rct, antspat_rct
    
    def __eq__(self, __value: object) -> bool:
        if self.excite_typ != __value.excite_typ:
            return False
        other_eeps = __value._get_embedded_elements()
        for antnr, eep in enumerate(self._get_embedded_elements()):
            if eep != other_eeps[antnr]:
                return False
        return True


class EEP_SC(EEPdata):
    def __init__(self, eep_sc, admittances_arr, voltage_excite=1.0):
        # Set up super, EEPdata, stuff first:
        super().__init__(eep_sc, admittances_arr, excite_typ='SC',
                         adm_or_imp_load=None, excite_val=voltage_excite)
        # Set up EEP_SC specific
        self.admittances = admittances_arr
        self.voltage_excite = voltage_excite
    
    def transform_to(self, excite_typ, excite_val=1., adm_load=None):
        """
        Transform this SC EEP to OC or NO
        
        Parameters
        ----------
        excite_typ: str
            Excitation type to transform to; can be: 'OC' or 'NO'.
        excite_val: float
            Value of the excitation voltage or current
        adm_load: array
            Load admittance for transform to 'NO', in Mhos.
            `None` (default) for transform to 'OC'.
            If your is given as an impedance instead of an admittance,
            the set this argument to 1/impedance. 
        """
        if excite_typ == self.excite_typ:
            return deepcopy(self)
        # _ee is placeholder var: gets overwritten before returning when
        # method .set_antspats_arr() is called
        _ee = deepcopy(self._get_embedded_elements())
        imp_arr = self.get_impedances()
        ap_SC = self.get_antspats_arr()
        # Note: antspats arr shape = (nrant, nrfreq, nrtheta, nrphi, nrpol)
        #       so ant-axis (0) should be moved to end and the end extended
        #       by one dimension so that it appears as stack of column vectors:
        #           antspat_SC[freq,theta,phi,pol, ants, 1] ...
        ap_SC = np.expand_dims(np.moveaxis(ap_SC, [0], [-1]), -1)
        # Normalize antspats_SC with excite voltages and current:
        ap_SC_0 = ap_SC * excite_val / self.voltage_excite
        if excite_typ == 'OC':
            eepdat_tr = EEP_OC(_ee, np.copy(imp_arr), excite_val)
        #       ... so it can be matmul (@) with e.g.
        #           imp_arr[freqs, np.newaxis, newaxis, np.newaxis, ants, ants]
        #       after expanding the axes 1,2,3:
            _imp_arr_ext = np.expand_dims(imp_arr, axis=(1,2,3))
            # Warnick2021 eq. 7
            antspat_tr = _imp_arr_ext @ ap_SC_0
        elif excite_typ == 'TH':
            raise NotImplementedError('Transform from SC -> TH not implemented')
        elif excite_typ == 'NO':
            if np.isscalar(adm_load):
                adm_load = adm_load*np.identity(imp_arr.shape[-1])
            elif adm_load.ndim == 1:
                adm_load = (np.expand_dims(adm_load, axis=(1,2,3,4,5))
                            * np.identity(imp_arr.shape[-1]))
            # Create new EEP_NO object to hold results to be return:ed
            adm_arr = np.copy(self.get_admittances())
            eepdat_tr = EEP_NO(_ee, adm_arr, adm_load, excite_val)
            _adm_arr_ext = np.expand_dims(adm_arr, axis=(1,2,3))
            # Warnick2021 eq. 6
            antspat_tr = np.linalg.inv(adm_load + _adm_arr_ext) @ ap_SC_0
        antspat_tr = np.moveaxis(antspat_tr.squeeze(-1), [-1], [0])
        eepdat_tr.set_antspat_arr(antspat_tr)
        return eepdat_tr


class EEP_OC(EEPdata):
    def __init__(self, eep_oc, impedances_arr, current_excite=1.0):
        # Set up super, EEPdata, stuff:
        super().__init__(eep_oc, impedances_arr, excite_typ='OC',
                         adm_or_imp_load=None, excite_val=current_excite)
        self.impedances = impedances_arr
        self.current_excite = current_excite
    
    def transform_to(self, excite_typ, excite_val=1., imp_load=None):
        if excite_typ == self.excite_typ:
            return deepcopy(self)
        _ee = deepcopy(self._get_embedded_elements())
        ap_OC = self.get_antspats_arr()
        ap_OC = np.expand_dims(np.moveaxis(ap_OC, [0], [-1]), -1)
        ap_OC_0 = ap_OC * excite_val / self.current_excite
        if excite_typ == 'SC':
            adm_arr = np.copy(self.get_admittances())
            eepdat_tr = EEP_SC(_ee, adm_arr, excite_val)
            _adm_arr_ext = np.expand_dims(adm_arr, axis=(1,2,3))
            # Warnick2021 eq. 7
            antspat_tr = _adm_arr_ext @ ap_OC_0
        elif excite_typ == 'TH':
            imp_arr = np.copy(self.get_impedances())
            if np.isscalar(imp_load):
                imp_load = imp_load*np.identity(imp_arr.shape[-1])
            eepdat_tr = EEP_TH(_ee, imp_arr, imp_load, excite_val)
            _imp_arr_ext = np.expand_dims(imp_arr, axis=(1,2,3))
            # Warnick2021 eq. 4
            antspat_tr = np.linalg.inv(imp_load + _imp_arr_ext) @ ap_OC_0
        elif excite_typ == 'NO':
            raise NotImplementedError('Transform from OC -> NO not implemented')
        antspat_tr = np.moveaxis(antspat_tr.squeeze(-1), [-1], [0])
        eepdat_tr.set_antspat_arr(antspat_tr)
        return eepdat_tr


class EEP_NO(EEPdata):
    def __init__(self, eep_no, adm_arr, adm_load, current_excite=1.0):
        # Set up super, EEPdata, stuff:
        super().__init__(eep_no, adm_arr, excite_typ='NO',
                         adm_or_imp_load=adm_load, excite_val=current_excite)
        self.admittances = adm_arr
        self.adm_load = adm_load
        self.current_excite = current_excite

    def transform_to(self, excite_typ, excite_val=1.):
        if excite_typ == self.excite_typ:
            return deepcopy(self)
        _ee = deepcopy(self._get_embedded_elements())
        adm_arr = self.get_admittances()
        ap_NO = self.get_antspats_arr()
        ap_NO = np.expand_dims(np.moveaxis(ap_NO [0], [-1]), -1)
        ap_NO_0 = ap_NO * excite_val / self.current_excite
        if excite_typ == 'SC':
            eepdat_tr = EEP_SC(_ee, np.copy(adm_arr), excite_val)
            _adm_arr_ext = np.expand_dims(adm_arr, axis=(1,2,3))
            # Warnick2021 eq. 6
            antspat_tr = np.linalg.inv(self.adm_load + _adm_arr_ext) @ ap_NO_0
        elif excite_typ == 'OC':
            raise NotImplementedError('Transform from NO -> OC not implemented')
        elif excite_typ == 'TH':
            raise NotImplementedError('Transform from NO -> TH not implemented')
        antspat_tr = np.moveaxis(antspat_tr.squeeze(-1), [-1], [0])
        eepdat_tr.set_antspat_arr(antspat_tr)
        return eepdat_tr


class EEP_TH(EEPdata):
    def __init__(self, eep_th, imp_arr, imp_load, voltage_excite=1.0):
        # Set up super, EEPdata, stuff:
        super().__init__(eep_th, imp_arr, excite_typ='TH',
                         adm_or_imp_load=imp_load, excite_val=voltage_excite)
        self.impedances = imp_arr
        self.imp_load = imp_load
        self.voltage_excite = voltage_excite

    def transform_to(self, excite_typ, excite_val=1.):
        if excite_typ == self.excite_typ:
            return deepcopy(self)
        _ee = deepcopy(self._get_embedded_elements())
        imp_arr = self.impedances
        ap_TH = self.get_antspats_arr()
        ap_TH = np.expand_dims(np.moveaxis(ap_TH, [0], [-1]), -1)
        ap_TH_0 = ap_TH * excite_val / self.current_excite
        if excite_typ == 'SC':
            raise NotImplementedError('Transform from TH -> SC not implemented')
        elif excite_typ == 'OC':
            eepdat_tr = EEP_OC(_ee, np.copy(imp_arr), excite_val)
            _imp_arr_ext = np.expand_dims(imp_arr, axis=(1,2,3))
            # Warnick2021 eq. 4
            antspat_tr = (self.imp_load + _imp_arr_ext) @ ap_TH_0
        elif excite_typ == 'NO':
            raise NotImplementedError('Transform from TH -> NO not implemented')
        antspat_tr = np.moveaxis(antspat_tr.squeeze(-1), [-1], [0])
        eepdat_tr.set_antspat_arr(antspat_tr)
        return eepdat_tr
    

class EELdata(EEPdata):
    def __init__(self, eels, adm_or_imp, excite_typ=None, adm_or_imp_load=None):
        self.eels = eels
        self.adm_or_imp = adm_or_imp
        self.excite_typ = excite_typ
        self.adm_or_imp_load = adm_or_imp_load

    def _get_embedded_elements(self):
        return self.eels
    
    def area_eff(self):
        """\
        Compute the effective area

        Returns
        -------
        area_effs : array
            Effective area with the same shape as the self.eels
        """
        if self.excite_typ != 'TH' and self.excite_typ != 'NO':
            raise NotImplementedError(
                f'Excite type is {self.excite_typ} but only TH and NO loading')
        area_effs = []
        load_adm_or_imp = self.adm_or_imp_load
        if load_adm_or_imp.ndim == 2:
            load_adm_or_imp_ = np.diagonal(self.adm_or_imp_load,
                                           axis1=-2, axis2=-1)
        else:
            load_adm_or_imp_ = np.squeeze(
                np.moveaxis(
                    np.diagonal(self.adm_or_imp_load, axis1=-2, axis2=-1),
                -1, 0),
            -1)
        for eel, aoi_l in zip(self.eels, load_adm_or_imp_):
            a_e_cmplx_un = ETA0 / 2 * (np.abs(eel.f_tht)**2
                                       +np.abs(eel.f_phi)**2)
            if self.excite_typ == 'TH':
                a_e_cmplx = 1/aoi_l * a_e_cmplx_un
            elif self.excite_typ == 'NO':
                a_e_cmplx = aoi_l * a_e_cmplx_un
            a_e = np.real(a_e_cmplx)
            area_effs.append(a_e)
        area_effs = np.asarray(area_effs)
        return area_effs


class TaggedGroup:

    def __init__(self):
        self.parts = {}
        self._tag_nr = 0
        self._group_id = ''     # Should match one of StructureModel dict
                                # attribute groups keys.
    
    @classmethod
    def _uniq_name(cls, basename, namesdict):
        genericnames = [gn for gn in namesdict if gn.startswith(basename)]
        idx = 0
        while True:
            name = basename+str(idx)
            if name not in genericnames: break
            idx +=1
        name = basename+str(idx)
        return name

    def make_wire(self, point_src, point_dst, thickness, name='', nr_seg=None):
        wire = Wire(point_src, point_dst, thickness, nr_seg)
        if not name:
            name = self._uniq_name('_part_', self.parts)
        self.parts[name] = wire
        return self
    
    def make_transformation(self, rox, roy, roz, xs, ys, zs, nrpt,
                            itgi=0, its=0):
        pass

    def get_ports(self, port_name=None):
        __ports = {}
        for _partid in self.parts:
             __port = self.parts[_partid].port
             if __port:
                 __ports[__port.name] = __port
        if port_name is None:
            return __ports
        else:
            return __ports[port_name]

    def _port_part(self, port_name):
        for __partid in self.parts:
            try:
                part_port_name = self.parts[__partid].port.name
            except AttributeError:
                part_port_name = None
            if port_name == part_port_name:
                return __partid
        raise KeyError(f"Port '{port_name}' does not exist")

    def _assign_port_segs(self):
        """Assign Port Segments

        Go through all ports in this group, find the segment to attach to and
        assign that segment to be the segment that gets excited together with
        the attributes of the ports attribute.
        """
        for portid in self.get_ports():
            fp = self.get_ports(portid).fractional_position
            partid = self._port_part(portid)
            prmtrc_seg_in_part = self.parts[partid].parametric_seg(fp)
            # Calculate segment offset within group
            offset_seg = 0
            for pid in self.parts:
                if pid == partid: break
                offset_seg += self.parts[pid].nr_seg
            self.get_ports(portid).ex_seg = offset_seg + prmtrc_seg_in_part

    def part_lengths(self):
        lengths = {}
        for pid in self:
            part = self.parts[pid]
            lengths[pid] = part.length()
        return lengths
    
    def total_length(self):
        return np.sum(list(self.part_lengths().values()))

    def get_nrsegments(self):
        return np.sum([self.parts[p].nr_seg for p in self.parts])

    def __iter__(self):
        for pid in self.parts:
            yield pid
        
    def __getitem__(self, part_name):
        return self.parts[part_name]
    
    def __setitem__(self, part_name, taggable):
        if type(taggable) == Wire:
            self.parts[part_name] = taggable
        elif type(taggable) == Transformations:
            pass
        elif type(taggable) == Port:
            raise DeprecationWarning('Setting port to part is discouraged')
            # Check that this group has the desired part
            if not self.parts.get(part_name):
                raise KeyError(
                    f"TaggedGroup '{self._group_id}' has no part '{part_name}'")
            if not taggable.name:
                taggable.name = self._uniq_name('_port_', self.get_ports())
            self.parts[part_name].port = taggable


class StructureModel:

    def __init__(self, name='Model_'):
        self.name = name
        self.groups = {}
        self.executionblocks = {}
        self.comments = []
        self.ground = None
        self._last_base_tag_nr = 0
    
    def set_commentline(self, comment):
        self.comments.append(comment)

    def set_ground(self, grnd=Ground(1, 0, 0, 0), gpflag=1):
        if grnd is not None:
            self.ground = {'gpflag': gpflag, 'grnd': grnd}

    def _assign_tags_base(self, exclude_groups=None):
        """\
        Assign sequential tag nrs to all groups except those in exclude group
        which by default is none 
        """
        last_tag_nr = self._last_base_tag_nr
        # Set tags for non element groups
        exclude_groups = {} if exclude_groups is None else exclude_groups
        nonelemgrps = set(self.groups)-set(exclude_groups)
        inc = 1
        for last_tag_nr, gid in enumerate(nonelemgrps, start=last_tag_nr+inc):
            self.groups[gid]._tag_nr = last_tag_nr
        self._last_base_tag_nr = last_tag_nr

    def nrsegs_hints(self, min_seg_perlambda, max_frequency):
        nrsegs = {}
        max_frequency = max_frequency * 1e6
        _min_lambda = 3e8/max_frequency
        for gid in self.groups:
            plens = self.groups[gid].part_lengths()
            for pid in plens:
                p_lenlambda = plens[pid]/_min_lambda
                nrsegs[(gid,pid)] = round(p_lenlambda*min_seg_perlambda)
        return nrsegs
    
    def segmentalize(self, min_seg_perlambda, max_frequency):
        if min_seg_perlambda is None:
            min_seg_perlambda = 10
        max_nrseg = self.nrsegs_hints(min_seg_perlambda, max_frequency)
        for gid, pid in max_nrseg:
            self.groups[gid].parts[pid].nr_seg = max_nrseg[(gid, pid)]
        self._assign_tags_base()
        return max_nrseg
    
    def seglamlens(self, freqsteps=None):
        """Compute Segment Length in Units of lambda

        Diagnostic to check segment lengths per wavelength.
        Recommendations from https://www.nec2.org/part_3/secii.html
        are to have segment length between 0.001 and 0.1 lambda
        (i.e. between 10 to 1000 segs per lambda).

        This method computes for all parts of the structure model over
        all frequencies in the frequency execution or freqsteps passed
        as argument.
        """
        if freqsteps == None:
            try:
                eblst = next(reversed(self.executionblocks.values()))
                freqsteps = eblst.freqsteps
            except:
                raise RuntimeError('No freqsteps defined')
        lams = 3e8/(1e6*np.asarray(freqsteps.aslist()))
        lams_per_seg = {}
        for gid, pid in self:
            _part = self.groups[gid].parts[pid]
            nr_seg_part = _part.nr_seg
            len_part = _part.length()
            segs_per_len = nr_seg_part/len_part
            segs_per_lam = segs_per_len * lams
            lams_per_seg[(gid,pid)] = 1 / segs_per_lam
        return lams_per_seg
    
    def segthinness(self):
        """Compute Segment Thinness

        Diagnostic to see if thin-wire extension kernel is needed.
        Recommendations from https://www.nec2.org/part_3/secii.html
        are to have seglen/radius > 8, for thin-wire kernel option,
        and seglen/radius < 2, for extended thin-wire kernel option.

        Like seglamlens(), this method computes for all parts of the structure
        model over all frequencies in the frequency execution or freqsteps
        passed as argument.
        """
        lams_per_seg = self.seglamlens()
        seg_thinness = {}
        for gidpid in lams_per_seg:
            gid, pid = gidpid
            _part = self.groups[gid].parts[pid]
            nr_seg_part = _part.nr_seg
            len_part = _part.length()
            seglen = len_part/nr_seg_part
            a = _part.radius
            thinness = seglen/a
            seg_thinness[(gid,pid)] = thinness
        return seg_thinness

    def _port_group(self, port_name):
        for gid in self.groups:
            if port_name in self.groups[gid].get_ports():
                return gid
        raise KeyError(f"Port '{port_name}' does not exist")

    def reset_port_srcs(self):
        """Remove all port excitations"""
        for base_gid in self.groups:
            _ports = self.groups[base_gid].get_ports()
            for _port in _ports:
                _pp = self.groups[base_gid]._port_part(_port)
                self.groups[base_gid][_pp].port.source = None

    def excite_port(self, port_id, voltage_src):
        """Attach a voltage source to port"""
        elem_nr = None
        if type(port_id) == str:
            port_name = port_id
        elif type(port_id) == tuple:
            elem_nr, port_name = port_id
        else:
            raise KeyError(f"Port '{port_id}' does not exist")
        base_gid = self._port_group(port_name)
        self.groups[base_gid].get_ports(port_name).source = voltage_src
        if elem_nr is not None:
            self.excited_elements.append(port_id)

    def add_executionblock(self, name, executionblock , reset=False):
        if reset:
            self.executionblocks = {}
        self.executionblocks[name] = executionblock
    
    def _create_geom_groups(self, d, subgroup_ids):
        subgroups = {gid: self.groups[gid] for gid in subgroup_ids}
        for gid in subgroups:
            for pid in subgroups[gid]:
                tag_nr = subgroups[gid]._tag_nr
                a_part = subgroups[gid].parts[pid]
                nr_seg = a_part.nr_seg
                if type(a_part) == Wire:
                    d.append_card('GW', tag_nr, nr_seg,
                            *a_part.point_src, *a_part.point_dst,
                            a_part.radius)

    def _create_geom_exclusive_groups(self, d, subgroup_ids):
        pass

    def _create_excite_groups(self, d, subgroup_ids, _exciteports):
        subgroups = {gid: self.groups[gid] for gid in subgroup_ids}
        exciteportsdct = dict(_exciteports)
        excited_ports_created = []
        for gid in subgroups:
            tag_nr = subgroups[gid]._tag_nr
            subgroups[gid]._assign_port_segs()
            for portid in subgroups[gid].get_ports():
                port = subgroups[gid].get_ports(portid)
                if not port.source:
                    if port.name not in exciteportsdct: continue
                    port.source = exciteportsdct[port.name]
                excited_ports_created.append(portid)
                ex_seg = port.ex_seg
                ex_type = port.source.nec_type()
                if ex_type == 0:
                    I2, I3 = tag_nr, ex_seg
                    print_max_rel_admittance_mat = 0
                    print_inp_imp = 1
                    I4 = int(
                        f"{print_max_rel_admittance_mat}{print_inp_imp}")
                voltage =  port.source.value
                if ex_type == 0:
                    F1, F2 = voltage.real, voltage.imag
                d.append_card('EX', ex_type, I2, I3, I4,
                                F1, F2, 0., 0., 0., 0.)
        return excited_ports_created

    def _create_excite_exclusive_groups(self, d, _exciteports, exciteports_grp):
        pass

    def as_neccards(self, exclude_groups=None):
        """\
        Return a Deck() object that corresponds to this StructureModel() object
        """
        d = Deck()
        
        # Comments
        for comment in self.comments:
            d.append_card('CM', ' '+comment)
        d.append_card('CE', '')
        exclude_groups = {} if exclude_groups is None else exclude_groups
        nonelemgrp = set(self.groups)-set(exclude_groups)

        # Structure Geometry for non element groups
        self._create_geom_groups(d, nonelemgrp)
        # Structure Geometry for element groups
        # # Create initial group to move
        self._create_geom_exclusive_groups(d, exclude_groups)

        # End Geometry
        gpflag = 0
        if self.ground:
            gpflag = self.ground['gpflag']
        d.append_card('GE', gpflag)

        # Program Control (loop over self.executionblocks)
        for _exblk in self.executionblocks.values():
            _freqsteps = _exblk.freqsteps
            _exciteports = _exblk.exciteports
            _radpat = _exblk.radpat

            # Extended Thin-Wire Kernel option?
            if _exblk.ext_thinwire:
                d.append_card('EK', 1)
            
            # Ground
            if self.ground:
                d.append_card('GN', *(self.ground['grnd'].astuple()))

            if _freqsteps:
            # Frequency
                I1, I2 = _freqsteps.to_nec_type(), _freqsteps.nrsteps
                F1, F2 = _freqsteps.start, _freqsteps.incr
                d.append_card('FR', I1, I2, F1, F2)

            # Excitations
            # ... non element group
            exciteports_grp = \
                self._create_excite_groups(d, nonelemgrp, _exciteports)
            # ... element group
            self._create_excite_exclusive_groups(d, _exciteports,
                                                    exciteports_grp)

            # Cards that trigger execution of NEC2 engine 
            if _radpat:
                d.append_card('RP', *astuple(_radpat))
            else:
                d.append_card('XQ', 0)

        # END
        d.append_card('EN', 0)
        return d
    
    def get_necout(self, eb, save_necfile=False, eb_id_suffix=''):
            self.add_executionblock('eb'+eb_id_suffix, eb, reset=True)
            _deck = self.as_neccards()
            if save_necfile:
                _deck.save_necfile(self.name+eb_id_suffix)
            freqs = eb.freqsteps.aslist()
            for nec_context in _deck.exec_pynec():
                ef_vert = []
                ef_hori = []
                voltages = []
                currents = []
                impedances = []
                for f in range(len(freqs)):
                    # Input (excitation) parameters
                    inp_parms = nec_context.get_input_parameters(f)
                    # ##frequency = inp_parms.get_frequency()
                    voltages.append(inp_parms.get_voltage())
                    currents.append(inp_parms.get_current())
                    impedances.append(inp_parms.get_impedance())

                    # Radiation pattern
                    radpat_out = nec_context.get_radiation_pattern(f)
                    # Coordinates theta,phi are the same for all frequecies,
                    # but easiest to just get it for each freq spec.
                    if radpat_out:
                        thetas = radpat_out.get_theta_angles()
                        phis = radpat_out.get_phi_angles()
                        # Fields
                        ef_vert_fr = radpat_out.get_e_theta()
                        ef_vert_fr = ef_vert_fr.reshape(
                                                (len(phis), len(thetas))).T
                        ef_hori_fr = radpat_out.get_e_phi()
                        ef_hori_fr = ef_hori_fr.reshape(
                                                (len(phis), len(thetas))).T
                        ef_hori.append(ef_hori_fr)
                        ef_vert.append(ef_vert_fr)
                    else:
                        thetas = None
                        phis = None
                necout = NECout(freqs, thetas, phis, np.array(ef_vert),
                                np.array(ef_hori), inp_V=np.array(voltages),
                                inp_I=np.array(currents),
                                inp_Z=np.array(impedances))
            return necout, nec_context

    def calc_eep_SC(self, eb, ref_port_nr=0, save_necfile=False):
            necout, _ = self.get_necout(eb, save_necfile)
            # Use impedance of excited reference port number ref_port_nr
            adm = 1/necout.inp_Z[:, ref_port_nr, np.newaxis, np.newaxis]
            eep_sc = EEP_SC([necout], adm, eb.exciteports[ref_port_nr][1].value)
            return eep_sc

    def __getitem__(self, group_id):
        if type(group_id) == str:
            if group_id not in self.groups:
                # New group
                if not group_id:
                    group_id = TaggedGroup._uniq_name('_group_', self.groups)
                tg = TaggedGroup()
                tg._group_id = group_id
                self.groups[group_id] = tg
            else:
                tg = self.groups[group_id]
        elif type(group_id) == tuple:
            tg = self.groups[group_id[1]]
        return tg
    
    def __setitem__(self, group_id, tag_group):
        tag_group._group_id = group_id
        self.groups[group_id] = tag_group

    def __iter__(self):
        for gid in self.groups:
            for pid in self.groups[gid].parts:
                yield gid, pid

    def __str__(self):
        indent = '    '
        out = ["Model: "+self.name]
        for g in self.groups:
            out.append(indent+'group: '+ str(g))
            _gs = self.groups[g]
            for p in _gs.parts:
                out.append(2*indent+"part: "+str(_gs.parts[p]))
        return '\n'.join(out)


class ArrayModel(StructureModel):

    def __init__(self, name='Model_'):
        super().__init__(name)
        self.element = []  # List of group names that make up element
        self.arr_delta_pos = [[]]
        self.elements_tags = [[]]  # Map element nr to its tags 
        self.excited_elements = []
        self._last_elem_tag_nr = self._last_base_tag_nr

    def _assign_tags_base(self):
        """\
        Overloaded method. Resets argument from default None to self.element
        """
        super()._assign_tags_base(exclude_groups=self.element)

    def _assign_tags_elem(self):
        """\
        Assign sequential tag nrs to array elements and set up element tags
        """
        nonelemgrps = set(self.groups)-set(self.element)
        if nonelemgrps is None:
            # Remove base group tags by starting from 0
            self._last_base_tag_nr = 0
        # Set tags for element group
        last_tag_nr = self._last_base_tag_nr
        inc = 10**len(str(last_tag_nr))
        #inc = 1  # REMOVE this test 
        elem_tags_start = last_tag_nr+inc
        for last_tag_nr, gid in enumerate(self.element, start=last_tag_nr+inc):
            self.groups[gid]._tag_nr = last_tag_nr
        self.elements_tags[0] = list(range(elem_tags_start, last_tag_nr+1))
        self._last_elem_tag_nr = last_tag_nr
        nr_elem_tags = len(self.elements_tags[0])
        for elem_nr in range(1, len(self.arr_delta_pos)):
            elem_tags_start += nr_elem_tags
            _ = range(elem_tags_start, elem_tags_start + nr_elem_tags)
            self.elements_tags.append(list(_))

    def arr_pos2arr_delta(self, array_positions):
        """Absolute array positions to array delta positions"""
        arr_delta_pos = []
        pos_from = [0., 0., 0.]
        for pos_to in array_positions:
            delta_pos = [pos_to[idx]-pos_from[idx] for idx in range(3)] 
            arr_delta_pos.append(delta_pos)
            pos_from = pos_to
        return arr_delta_pos
    
    def arr_delta2arr_pos(self, arr_delta_pos):
        """Array delta positions to absolute array positions"""
        array_positions = []
        resultant = [0., 0., 0.]
        for deltvec in arr_delta_pos:
            resultant = [resultant[idx]+deltvec[idx] for idx in range(3)]
            array_positions.append(resultant)
        return array_positions

    def arrayify(self, element, array_positions):
        self.element = element
        self.arr_delta_pos = self.arr_pos2arr_delta(array_positions)
        self._assign_tags_elem()
    
    def create_array(self, d):
        # Move 1st (base) element into pos0 (no tag increment)
        elem_tags_start = self.elements_tags[0][0]
        pos0 = self.arr_delta_pos[0]
        d.append_card('GM', 0, 0, 0., 0., 0.,
                        pos0[0], pos0[1], pos0[2], elem_tags_start)
        # Copy last element and move by delta_pos 
        #  (increment by nr of tags in base element after each move)
        nr_elem_tags = len(self.elements_tags[0])
        for dpos in self.arr_delta_pos[1:]:
            d.append_card('GM', nr_elem_tags, 1, 0., 0., 0.,
                            dpos[0], dpos[1], dpos[2], elem_tags_start)
            elem_tags_start += nr_elem_tags

    def _create_geom_exclusive_groups(self, d, subgroup_ids):
        super()._create_geom_groups(d, subgroup_ids)
        self.create_array(d)

    def _create_excite_exclusive_groups(self, d, _exciteports,
                                        exciteports_grp):
        # Check to see if non element groups are being excited
        if exciteports_grp:
            # This should probably be an exception but I can't
            # tell if there may be some interesting use case so
            # I leave it as a warning. 
            warnings.warn(f"Non embedded ports {exciteports_grp} excited so "
                          "ordinary embedded calculations will be incorrect. "
                          "Remove all port excitation values in model.")
        _nonelem_ex_ports = []
        _elem_ex_ports = []
        self.reset_port_srcs()
        for ep in _exciteports:
            self.excite_port(*ep)
            if type(ep[0]) == str:
                _nonelem_ex_ports.append(ep)
            elif type(ep[0]) == tuple:
                _elem_ex_ports.append(ep)
        for egname in self.element:
            self.groups[egname]._assign_port_segs()
        for _portid, __vs in _elem_ex_ports:
            eid, pnm = _portid
            gid = self._port_group(pnm)
            element_tag = self.elements_tags[eid][self.element.index(gid)]
            port = self.groups[gid].get_ports(pnm)
            if not port.source: continue
            ex_seg = port.ex_seg
            ex_type = port.source.nec_type()
            if ex_type == 0:
                I2, I3 = element_tag, ex_seg
                print_max_rel_admittance_mat = 0
                print_inp_imp = 1
                I4 = int(
                    f"{print_max_rel_admittance_mat}{print_inp_imp}")
            voltage =  port.source.value
            if ex_type == 0:
                F1, F2 = voltage.real, voltage.imag
            d.append_card('EX', ex_type, I2, I3, I4,
                            F1, F2, 0., 0., 0., 0.)

    def as_neccards(self):
        return super().as_neccards(exclude_groups=self.element)

    def excite_1by1(self, eep_eb, save_necfile=False, print_prog=False):
        """\
        Excite elements one at a time to obtain embedded element properties

        This method excites one element in the array at a time, leaving all
        other non-excited elements with their nominal loading.
        Since NEC2 is a MOM code, the excitation should be a voltage over
        a conducting segment, and this makes the EE procedure a short-cicuit
        EE excitation and the patterns will be the SC EEPs.
        
        Parameters
        ----------
        eep_eb : ExecutionBlock
            The execution block to used to compute the EEPs. Should have at
            least one excited port with an arbitary voltage value.
        save_nec : bool
            Save as a NEC file
        print_prog : bool
            Print progress by printing to screen the embedded element being
            excited.
        
        Returns
        -------
        results : EEPdata
            The EEP data, or specifically a EEP_SC() object.
        """
        _frq_cntr_step = eep_eb.freqsteps
        freqs = _frq_cntr_step.aslist()
        _exciteport_name, _vltsrc = eep_eb.exciteports
        _rad_pat = eep_eb.radpat
        nr_ants = len(self.arr_delta_pos)
        _eep_sc = []
        _admittances = np.zeros((len(freqs), nr_ants, nr_ants), complex)
        sc = StructureCurrents(freqs, nr_ants)
        for antnr in range(nr_ants):
            if print_prog:
                print(f'Exciting antenna {antnr}/{nr_ants}', end='\r',
                      flush=True)
            _prt_exc = ((antnr, _exciteport_name), _vltsrc)
            _xb = ExecutionBlock(_frq_cntr_step, [_prt_exc], _rad_pat,
                                 ext_thinwire=eep_eb.ext_thinwire)
            _necout, nec_context = super().get_necout(_xb, save_necfile,
                                                      eb_id_suffix=str(antnr))
            _eep_sc.append(_necout)
            for f in range(len(freqs)):
                # Get structure currents
                _sc_f = nec_context.get_structure_currents(f)
                _currents = _sc_f.get_current()
                _sc_segtags = _sc_f.get_current_segment_tag()
                _sc_segnums = _sc_f.get_current_segment_number()
                sc.currents = _currents
                sc.set_segtags(_sc_segtags)
                sc.set_segnums(_sc_segnums)

                # Find mutual-impedances
                admittances_T = []
                gid = self._port_group(_exciteport_name)
                port = self.groups[gid].get_ports(_exciteport_name)
                elemgrpidx = self.element.index(gid)
                for _antnr_j in range(nr_ants):
                    ex_tag = self.elements_tags[_antnr_j][elemgrpidx]
                    ex_seg = None
                    if port.source:
                        ex_seg = port.ex_seg
                    cur_port = sc.get_current(ex_tag, ex_seg)
                    admittances_T.append(cur_port / port.source.value)
                _admittances[f,:,antnr] = np.array(admittances_T)
        print() if print_prog else None
        results = EEP_SC(_eep_sc, _admittances, _vltsrc.value)
        return results

    def calc_steering_vector(self, eep_eb):
        """Calculate steering vector for array

        Returns
        -------
        steering_vectors : (nant, nfr, nth, nph) shaped array
            Steering vectors, i.e. the vector [exp(j*k_i*r_{la})].T
            for wavevector k_i (given by direction cosines times wavenumber)
            and the position vector r_l for all antennas a.
        """
        khat = eep_eb.radpat.as_khat()
        pos = self.arr_delta2arr_pos(self.arr_delta_pos)
        pos = np.array(pos)  # pos.shape = (nant, xyz)
        phases_hat = np.matmul(khat, pos.T)  # khat[nth,nph,xyz] pos[nant,xyz]
                                            # phase_hat.shape = (nth, nph, nant)
        _freqs = eep_eb.freqsteps.aslist()
        k = 2*np.pi/3e2*np.array(_freqs)  # k=2pi*freq/c, shape = (nfrq,)
        phases = k[:, np.newaxis, np.newaxis, np.newaxis] * phases_hat
        steering_vectors = np.exp(+1j*phases)  # sv[nfrq,nth,nph,nant]
        steering_vectors = np.moveaxis(steering_vectors, -1, 0)
        return steering_vectors

