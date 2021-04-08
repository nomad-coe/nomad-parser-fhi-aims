#############################################################################
#
# Script modified by Keith T. Butler to allow for command line modification of y-axis range with 'python scriptname.py -y lowerlim upperlim
#
#############################################################################
#
#
#
#  Script to plot band structure and DOS calculated with FHI-aims. Requires the control.in/geometry.in as well as the output
#  of the calculation to be in the same directory ...
#
#  To achieve labelling of the special points along the band structure plot, add two arguments to the "output band"
#  command in the control.in, using the following syntax:
#
#  output band <start> <end> <npoints> <starting_point_name> <ending_point_name>
#
#  Example: to plot a band with 100 points from Gamma to half way along one of the reciprocal lattice vectors, write (in control.in)
#
#  output band 0.0 0.0 0.0 0.5 0.0 0.0 100 Gamma <End_point_name>
#

from pylab import *
from numpy.linalg import *
from numpy import dot,cross,pi

import os,sys
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-y", "--ylim",
                  action="store", type="float", dest="ylim", nargs=2,
                  help="Y limits, in the form 0 1. Default = ene_min ene_max")
(options, args) = parser.parse_args()
ylim = options.ylim
print(ylim)



########################

print("Plotting bands for FHI-aims!")
print("============================")

print("Reading lattice vectors from geometry.in ...")

latvec = []

with open("geometry.in", "r") as fin:
    for line in fin:
        line = line.split("#")[0]
        words = line.split()
        if len(words) == 0:
            continue
        if words[0] == "lattice_vector":
            if len(words) != 4:
                raise Exception("geometry.in: Syntax error in line '"+line+"'")
            latvec += [ [float(x) for x in words[1:4]] ]

if len(latvec) != 3:
    raise Exception("geometry.in: Must contain exactly 3 lattice vectors")

latvec = asarray(latvec)

print("Lattice vectors:")
for i in range(3):
    print(latvec[i,:])

#Calculate reciprocal lattice vectors
rlatvec = []
volume = (np.dot(latvec[0,:],np.cross(latvec[1,:],latvec[2,:])))
rlatvec.append(array(2*pi*cross(latvec[1,:],latvec[2,:])/volume))
rlatvec.append(array(2*pi*cross(latvec[2,:],latvec[0,:])/volume))
rlatvec.append(array(2*pi*cross(latvec[0,:],latvec[1,:])/volume))
rlatvec = asarray(rlatvec)

#rlatvec = inv(latvec) Old way to calculate lattice vectors
print("Reciprocal lattice vectors:")
for i in range(3):
    print(rlatvec[i,:])

########################

print("Reading information from control.in ...")

PLOT_BANDS = False
PLOT_DOS = False
PLOT_DOS_SPECIES = False
PLOT_DOS_ATOM = False

species = []

max_spin_channel = 1
band_segments = []
band_totlength = 0.0 # total length of all band segments

with open("control.in", "r") as fin:
    for line in fin:
        words = line.split("#")[0].split()
        nline = " ".join(words)

        if nline.startswith("spin collinear"):
            max_spin_channel = 2

        if nline.startswith("output band "):
            if len(words) < 9 or len(words) > 11:
                raise Exception("control.in: Syntax error in line '"+line+"'")
            PLOT_BANDS = True
            start = array([float(x) for x in words[2:5]])
            end = array([float(x) for x in words[5:8]])
            length = norm(dot(rlatvec,end) - dot(rlatvec,start))
            band_totlength += length
            npoint = int(words[8])
            startname = ""
            endname = ""
            if len(words)>9:
                startname = words[9]
            if len(words)>10:
                endname = words[10]
            band_segments += [ (start,end,length,npoint,startname,endname) ]

        if nline.startswith("output dos"):
            PLOT_DOS = True

        if nline.startswith("output species_proj_dos"):
            PLOT_DOS = True
            PLOT_DOS_SPECIES = True

        if nline.startswith("output atom_proj_dos"):
            PLOT_DOS = True
            PLOT_DOS_ATOM = True

        if nline.startswith("species"):
            if len(words) != 2:
                raise Exception("control.in: Syntax error in line '"+line+"'")
            species += [ words[1] ]

#######################

if PLOT_BANDS and PLOT_DOS:
    ax_bands = axes([0.1,0.1,0.6,0.8])
    ax_dos = axes([0.72,0.1,0.18,0.8],sharey=ax_bands)
    ax_dos.set_title("DOS")
    setp(ax_dos.get_yticklabels(),visible=False)
    ax_bands.set_ylabel("E [eV]")
    PLOT_DOS_REVERSED = True
elif PLOT_BANDS:
    ax_bands = subplot(1,1,1)
elif PLOT_DOS:
    ax_dos = subplot(1,1,1)
    ax_dos.set_title("DOS")
    PLOT_DOS_REVERSED = False

#######################

if PLOT_BANDS:
    print("Plotting %i band segments..."%len(band_segments))

    ax_bands.axhline(0,color=(1.,0.,0.))

    prev_end = band_segments[0][0]
    distance = band_totlength/30.0 # distance between line segments that do not coincide

    iband = 0
    xpos = 0.0
    labels = [ (0.0,band_segments[0][4]) ]

    for start,end,length,npoint,startname,endname in band_segments:
        iband += 1

        if any(start != prev_end):
            xpos += distance
            labels += [ (xpos,startname) ]

        xvals = xpos+linspace(0,length,npoint)
        xpos = xvals[-1]

        labels += [ (xpos,endname) ]

        prev_end = end
        prev_endname = endname

        for spin in range(1,max_spin_channel+1):
            fname = "band%i%03i.out"%(spin,iband)
            idx = []
            kvec = []
            band_energies = []
            band_occupations = []
            with open(fname, "r") as fin:
                for line in fin:
                    words = line.split()
                    idx += [ int(words[0]) ]
                    kvec += [[float(x) for x in words[1:4]]]
                    band_occupations += [[float(x) for x in words[4::2]]]
                    band_energies += [[float(x) for x in words[5::2]]]
            assert(npoint) == len(idx)
            band_energies = asarray(band_energies)
            for b in range(band_energies.shape[1]):
                ax_bands.plot(xvals,band_energies[:,b],color=' br'[spin])

    tickx = []
    tickl = []
    for xpos,l in labels:
        ax_bands.axvline(xpos,color='k')
        tickx += [ xpos ]
        if len(l)>1:
            l = "$\\"+l+"$"
        tickl += [ l ]
    for x, l in zip(tickx, tickl):
        print("| %8.3f %s" % (x, repr(l)))

    ax_bands.set_xlim(labels[0][0],labels[-1][0])
    # ax_bands.set_xticks(tickx)
    # ax_bands.set_xticklabels(tickl)

#######################

if PLOT_DOS:
    def smoothdos(dos):
        dos = asarray(dos)
        # JW: Smoothing is actually done within FHI-aims...
        return dos
        for s in range(40):  # possible gaussian smoothing
            dos_old = dos
            dos = 0.5*dos_old
            dos[1:,...] += 0.25*dos_old[:-1,...]
            dos[0,...] += 0.25*dos_old[0,...]
            dos[:-1,...] += 0.25*dos_old[1:,...]
            dos[-1,...] += 0.25*dos_old[-1,...]
        return dos

    print("Plotting DOS")

#    if PLOT_DOS_ATOM:
#    elif PLOT_DOS_SPECIES:
    if PLOT_DOS_SPECIES:
        spinstrs = [ "" ]
        if max_spin_channel == 2:
            spinstrs = [ "_spin_up","_spin_down" ]

        energy = []
        tdos = []
        ldos = []
        maxdos = 0.0

        for s in species:
            val_s = []
            for ss in spinstrs:
                f = file(s+"_l_proj_dos"+ss+".dat")
                f.readline()
                mu = float(f.readline().split()[-2])
                f.readline()
                val_ss = []
                for line in f:
                    val_ss += [ map(float,line.split()) ]
                val_s += [ val_ss ]
            val_s = asarray(val_s).transpose(1,0,2)
            energy += [ val_s[:,:,0] ]
            tdos += [ smoothdos(val_s[:,:,1]) ]
            ldos += [ smoothdos(val_s[:,:,2:]) ]
            maxdos = max(maxdos,tdos[-1].max())
        for e in energy:
            for i in range(e.shape[1]):
                assert all(e[:,i] == energy[0][:,0])
        energy = energy[0][:,0]

    else:
        with open("KS_DOS_total.dat", "r") as f:
            f.readline()
            mu = float(f.readline().split()[-2])
            f.readline()
            energy = []
            dos = []
            if max_spin_channel == 1:
                for line in f:
                    e,d = line.split()
                    energy += [ float(e) ]
                    dos += [ (float(d),) ]
            else:
                for line in f:
                    e,d1,d2 = line.split()
                    energy += [ float(e) ]
                    dos += [ (float(d1),float(d2)) ]
            energy = asarray(energy)
            dos = smoothdos(dos)
            maxdos = dos.max()

    spinsgn = [ 1. ]
    if max_spin_channel == 2:
        spinsgn = [ 1.,-1. ]

    if PLOT_DOS_REVERSED:
        ax_dos.axhline(0,color='k',ls='--')
        ax_dos.axvline(0,color=(0.5,0.5,0.5))

        if PLOT_DOS_SPECIES:
            for sp in range(len(species)):
                for ispin in range(max_spin_channel):
                    ax_dos.plot(tdos[sp][:,ispin]*spinsgn[ispin],energy,linestyle='-',label='%s %s'%(species[sp],['up','down'][ispin]))
                    for l in range(ldos[sp].shape[2]):
                        ax_dos.plot(ldos[sp][:,ispin,l]*spinsgn[ispin],energy,linestyle='--',label='%s (l=%i) %s'%(species[sp],l,['up','down'][ispin]))
        else:
            for ispin in range(max_spin_channel):
                ax_dos.plot(dos[:,ispin]*spinsgn[ispin],energy,color='br'[ispin])

        ax_dos.set_xlim(array([min(spinsgn[-1],0.0)-0.05,1.05])*maxdos)
        if ylim:
            ax_dos.set_ylim(float(ylim[0]),float(ylim[1]))
        else:
            ax_dos.set_ylim(energy[0],energy[-1])
    else:
        ax_dos.axvline(0,color='k',ls='--')
        ax_dos.axhline(0,color=(0.5,0.5,0.5))

        if PLOT_DOS_SPECIES:
            for sp in range(len(species)):
                for ispin in range(max_spin_channel):
                    ax_dos.plot(energy,tdos[sp][:,ispin]*spinsgn[ispin],color='br'[ispin],linestyle='-',label='%s %s'%(species[sp],['up','down'][ispin]))
                    for l in range(ldos[sp].shape[2]):
                        ax_dos.plot(energy,ldos[sp][:,ispin,l]*spinsgn[ispin],color='br'[ispin],linestyle='--',label='%s (l=%i) %s'%(species[sp],l,['up','down'][ispin]))
        else:
            for ispin in range(max_spin_channel):
                ax_dos.plot(energy,dos[:,ispin]*spinsgn[ispin],color='br'[ispin])

        ax_dos.set_xlim(energy[0],energy[-1])
        ax_dos.set_ylim(array([min(spinsgn[-1],0.0)-0.05,1.05])*maxdos)
        ax_dos.set_xlabel(r"$\varepsilon - \mu$ (eV)")
    if PLOT_DOS_SPECIES:
        ax_dos.legend()
else:
    ax_bands.set_ylim(-20,20) # just some random default -- definitely better than the full range including core bands

#######################

def on_q_exit(event):
    if event.key == "q": sys.exit(0)
connect('key_press_event', on_q_exit)
show()
