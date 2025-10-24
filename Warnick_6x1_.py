"""\
Recreate Figs 2 & 3 from Warnick2021
"""
import matplotlib.pyplot as plt
import numpy as np
from nec2array import (ArrayModel, VoltageSource, FreqSteps, Wire,
                  ExecutionBlock, RadPatternSpec)

#### Build array element antenna
# Everything scaled to lambda so actual lambda is arbitrary
# but for simplicity set it to 1m. 
lambda_ = 1
p1 = (0., 0., -lambda_/2/2)
p2 = (0., 0., +lambda_/2/2)
l12 = (p1, p2)
wire_radius = 1e-6*lambda_

model_name = __file__.replace('.py','')
warnick6 = ArrayModel(model_name)
warnick6.set_commentline(warnick6.name)
warnick6.set_commentline('Author: T. Carozzi')
warnick6.set_commentline('Date: 2024-03-20')
warnick6['ant_Z']['w'] = Wire(*l12, wire_radius)
warnick6['ant_Z']['w'].add_port(0.5, 'the_port')

# Set up execution settings
nr_freqs = 1
frq_cntr = 3.0e8 * lambda_ /1e6
frq_cntr_step = FreqSteps('lin', nr_freqs, frq_cntr)
warnick6.segmentalize(201, frq_cntr)
_port_ex = ('the_port', VoltageSource(1.0))
nph = 100
rps = RadPatternSpec(thets=90., nph=nph, dph=360/nph, phis=0.)
eb_arr = ExecutionBlock(frq_cntr_step, _port_ex, rps)

#### Set up array of 6 elements with two different spacings
#### and reference antenna to nr. 2
nr_ants = 6
ref_ant = 2
frq_idx = 0
pats_OC = []
pats_SC = []
pows_act_OC = []
pows_act_SC = []
# The two spacings in units of lambda:
spacefacs =  [0.5, 0.2]
for spfac in spacefacs:
    d = spfac*lambda_
    arr_pos =[[antnr*d-(nr_ants-1)/2*d, 0., 0.] for antnr in range(nr_ants)]
    warnick6.arrayify(element=['ant_Z'], array_positions=arr_pos)
    eepSCdat = warnick6.excite_1by1(eb_arr)
    pow_act_SC = eepSCdat.get_pow_arr()[frq_idx, ref_ant, ref_ant]
    pows_act_SC.append(pow_act_SC)
    # Get OC from SC through transformation:
    eepOCdat = eepSCdat.transform_to('OC')
    pow_act_OC = (np.abs(eepOCdat.current_excite)**2
            *np.real(eepOCdat.impedances[frq_idx,ref_ant,ref_ant])/2.)
    pows_act_OC.append(pow_act_OC)
    _antspats = eepSCdat.get_antspats_arr()
    pats_SC.append(_antspats[:,0,0,:,:].squeeze())
    _antspats = eepOCdat.get_antspats_arr()
    pats_OC.append(_antspats[:,0,0,:,:].squeeze())

def dbi(efield, pow):
    eta = 377
    U = (np.abs(efield[:,0])**2 + np.abs(efield[:,1])**2)/(2*eta)
    dbi_ = 10*np.log10(U/pow*4*np.pi)
    return dbi_

#### Plot the results
phis = rps.as_thetaphis()[1]

fig, axs = plt.subplots(2,1, sharex=True)
for spidx in range(len(spacefacs)):
    pat_SC_dbi = dbi(pats_SC[spidx][ref_ant,:], pows_act_SC[spidx])
    axs[spidx].plot(phis, pat_SC_dbi, 'y-.')
    pat_OC_dbi = dbi(pats_OC[spidx][ref_ant,:], pows_act_OC[spidx])
    axs[spidx].plot(phis, pat_OC_dbi, 'r-.')
    axs[spidx].set_title(f'd={spacefacs[spidx]}*lambda')
    axs[spidx].set_xlabel('phi [deg]')
    axs[spidx].set_ylabel('Directivity [dBi]')
    axs[spidx].legend(['SC', 'OC (from SC)'])
plt.show()
