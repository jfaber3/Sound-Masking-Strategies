import numpy as np
import matplotlib.pyplot as plt
import code
from scipy.integrate import solve_ivp


def LP_filter(x, framerate, cutoff_f):
    N = len(x)
    xf = np.fft.fft(x)
    f = np.linspace(0.0, framerate/2, int(N/2))
    for i in range(len(f)):
        if f[i] > cutoff_f:
            xf[i] = 0.0 + 0.0j
            xf[len(xf)-i-1] = 0.0 + 0.0j
    return np.real(np.fft.ifft(xf))

def generate_FM_signal(N,  dt,  w,  w_dev):
    Dw = np.random.normal(0, w_dev, N)
    Dw = LP_filter(Dw, 1/dt, w/(2*np.pi))
    Dw *= w_dev/Dw.std()
    phase = np.zeros(N)
    for i in range(1, len(phase)):
        phase[i] = phase[i-1] + (w + Dw[i])*dt
    return phase

def PSD_avg_segs(x, framerate, num_segs):
    N = int(len(x)/num_segs)
    f = np.linspace(0.0, framerate/2, int(N/2))
    xff_avg = np.zeros(len(f))
    for i in range(num_segs):
        xf = np.fft.fft( x[i*N:(i+1)*N] )
        xff = (2.0/(N*framerate))*np.abs(xf[0:int(N/2)])**2
        xff_avg += xff/num_segs
    return f, xff_avg

def make_Hopf_dots_complex(F_full_complex):
    def Hopf_dots_complex(t, z, mu, w0):
        F_complex = np.interp(t, tt, F_full_complex)
        return (mu + w0*1j - abs(z)**2)*z + F_complex
    return Hopf_dots_complex



#Mask Functions

def gaussian_noise_mask(tot_power, w_mask, w_mask_dev, N, dt):
    mean, sdev = w_mask/(2*np.pi), w_mask_dev/(2*np.pi)
    if sdev == 0:
        sdev += 10**(-10)  #so it doesn't blow up
    f = np.linspace(0.0, 1/(2*dt), int(N/2))
    f_peak = f[np.argmin(np.abs(f-mean))]   #centering on nearest freq bin
    A = np.exp( -0.5*( (f-f_peak)/sdev )**2 )
    phi = np.random.rand(int(N/2))*2*np.pi  #random phases for all components
    xf_pos = (np.cos(phi          ) + 1.0j*np.sin(phi          )) * ((N/(2*dt))*A)**0.5
    yf_pos = (np.cos(phi - np.pi/2) + 1.0j*np.sin(phi - np.pi/2)) * ((N/(2*dt))*A)**0.5  #rotate 90 degrees for y
    xf = np.append(xf_pos, np.zeros(len(xf_pos), dtype='complex'))
    yf = np.append(yf_pos, np.zeros(len(yf_pos), dtype='complex'))
    x, y = np.real(np.fft.ifft(xf)), np.real(np.fft.ifft(yf))
    x *= ((tot_power/2)**0.5)/x.std()
    y *= ((tot_power/2)**0.5)/y.std()
    return x + 1j*y

def FM_linear_mask(tot_power, f_mod, a_mod, tt):  #linear ramp up
    W = 2*np.pi + a_mod*( 2*(  (f_mod*tt)%(1)  ) - 1) 
    phi = np.cumsum(W*(tt[1]-tt[0]))
    return np.sqrt(tot_power)*np.exp(1j*phi)

def AM_mask(tot_power, w_mod, a_mod, tt):
    w_mask = 1.01*(2*np.pi)
    assert a_mod <= np.sqrt(2*tot_power)
    a0 = np.sqrt(tot_power - 0.5*a_mod**2)
    return (a0 + a_mod*np.sin(w_mod*tt))*np.exp(1j*w_mask*tt)


np.random.seed(5)   #5
Ff = 0.3  #0.3
Fm = 1
mu = 0.1
w0 = 2*np.pi
wdp =  np.pi
z0 = [mu**0.5 + 0j]  #initial condition
num_cycles = 1000
pts_per_cycle = 100  #100
tt = np.linspace(0, num_cycles, num_cycles*pts_per_cycle)


mask_power = 9 *Ff**2
F_male   = Fm*np.exp(1j*1.5*w0*tt)
F_female = Ff*np.exp(1j*generate_FM_signal(len(tt), tt[1]-tt[0], w0, 0.2*w0))
F_mask   = gaussian_noise_mask(mask_power, w0, 0.1*w0, len(tt), tt[1]-tt[0])   #Ff*np.exp(1j*w0*1.2*tt)
#F_mask   = FM_linear_mask(mask_power, 0.1, 0.5*w0, tt)
#F_mask = AM_mask(mask_power, 0.3*w0, 0.3, tt)

#1 Hopf, no mask
sol1 = solve_ivp(make_Hopf_dots_complex(F_female), (0, num_cycles), z0, args=(mu, w0), t_eval=tt, method='RK45')
x1, y1 = np.real(sol1.y[0]), np.imag(sol1.y[0])

#1 Hopf with mask
sol1_m = solve_ivp(make_Hopf_dots_complex(F_female + F_mask), (0, num_cycles), z0, args=(mu, w0), t_eval=tt, method='RK45')
x1m, y1m = np.real(sol1_m.y[0]), np.imag(sol1_m.y[0])

#2 Hopf, no mask
sol_interm = solve_ivp(make_Hopf_dots_complex(F_female + F_male), (0, num_cycles), z0, args=(mu, w0), t_eval=tt, method='RK45')
sol2 = solve_ivp(make_Hopf_dots_complex(sol_interm.y[0]), (0, num_cycles), z0, args=(mu, wdp), t_eval=tt, method='RK45')
x2, y2 = np.real(sol2.y[0]), np.imag(sol2.y[0])

#2 Hopf with mask
sol_interm_m = solve_ivp(make_Hopf_dots_complex(F_female + F_male + F_mask), (0, num_cycles), z0, args=(mu, w0), t_eval=tt, method='RK45')
sol2_m = solve_ivp(make_Hopf_dots_complex(sol_interm_m.y[0]), (0, num_cycles), z0, args=(mu, wdp), t_eval=tt, method='RK45')
x2m, y2m = np.real(sol2_m.y[0]), np.imag(sol2_m.y[0])



#x2 = LP_filter(x2, 1/(tt[1]-tt[0]), w0/(2*np.pi))   #testing filter


num_seg = 20
f_male, xf_male     = PSD_avg_segs(np.real(F_male), pts_per_cycle, num_seg)
f_female, xf_female = PSD_avg_segs(np.real(F_female), pts_per_cycle, num_seg)
f_mask, xf_mask     = PSD_avg_segs(np.real(F_mask), pts_per_cycle,   num_seg)
f1,  xf1,              = PSD_avg_segs(x1,  pts_per_cycle, num_seg)
f1m, xf1m,             = PSD_avg_segs(x1m, pts_per_cycle, num_seg)
f2,  xf2,              = PSD_avg_segs(x2,  pts_per_cycle, num_seg)
f2m, xf2m,             = PSD_avg_segs(x2m, pts_per_cycle, num_seg)


fig = plt.figure(figsize=(5, 5))
plt.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.99, wspace=0.5, hspace=0.1)
ax1  = plt.subplot2grid((5, 3), (0, 0), rowspan=1, colspan=2)
ax2  = plt.subplot2grid((5, 3), (1, 0), rowspan=1, colspan=2)
ax3  = plt.subplot2grid((5, 3), (2, 0), rowspan=1, colspan=2)
ax4  = plt.subplot2grid((5, 3), (3, 0), rowspan=1, colspan=2)
ax5  = plt.subplot2grid((5, 3), (4, 0), rowspan=1, colspan=2)
ax11  = plt.subplot2grid((5, 3), (0, 2), rowspan=1, colspan=1)
ax12  = plt.subplot2grid((5, 3), (1, 2), rowspan=1, colspan=1)
ax13  = plt.subplot2grid((5, 3), (2, 2), rowspan=1, colspan=1)
ax14  = plt.subplot2grid((5, 3), (3, 2), rowspan=1, colspan=1)
ax15  = plt.subplot2grid((5, 3), (4, 2), rowspan=1, colspan=1)

alph = 1
clrs = ["black", "#00AAFF", "#FF7300", "#FF00FB"]  #female,   res 1Hopf,    res 2Hopf,  mask

ax1.plot(tt, np.real(F_female), color=clrs[0])
ax1.plot(tt, x1,                color=clrs[1],   alpha=alph)
ax2.plot(tt, np.real(F_female), color=clrs[0])
ax2.plot(tt, x2,                color=clrs[2],   alpha=alph)

ax3.plot(tt, np.real(F_mask),   color=clrs[3])

ax4.plot(tt, np.real(F_female), color=clrs[0])
ax4.plot(tt, x1m,               color=clrs[1],   alpha=alph)
ax5.plot(tt, np.real(F_female), color=clrs[0])
ax5.plot(tt, x2m,               color=clrs[2],   alpha=alph)


xrange = (num_cycles-30, num_cycles-5)
yrange = (-1.4, 1.4)
ax1.set_xlim(xrange)
ax2.set_xlim(xrange)
ax3.set_xlim(xrange)
ax4.set_xlim(xrange)
ax5.set_xlim(xrange)
ax1.set_ylim(yrange)
ax2.set_ylim(yrange)
ax3.set_ylim(yrange)
ax4.set_ylim(yrange)
ax5.set_ylim(yrange)

ax1.text(xrange[1]-5.5, -1.3, "mask off")
ax2.text(xrange[1]-5.5, -1.3, "mask off")
ax4.text(xrange[1]-5.5, -1.3, "mask on")
ax5.text(xrange[1]-5.5, -1.3, "mask on")

ax11.plot(f_female, xf_female,   color=clrs[0])
ax11.plot(f1, xf1,               color=clrs[1], alpha=alph)
ax12.plot(f_female, xf_female,   color=clrs[0])
ax12.plot(f2, xf2,               color=clrs[2], alpha=alph)

ax13.plot(f_mask, xf_mask,       color=clrs[3])

ax14.plot(f_female, xf_female,   color=clrs[0])
ax14.plot(f1, xf1m,              color=clrs[1], alpha=alph)
ax15.plot(f_female, xf_female,   color=clrs[0])
ax15.plot(f2, xf2m,              color=clrs[2], alpha=alph)

xrange = (0, 2)
yrange = (0.001, 5)
ax11.set_xlim(xrange)
ax11.set_ylim(yrange)
ax12.set_xlim(xrange)
ax12.set_ylim(yrange)
ax13.set_xlim(xrange)
ax13.set_ylim(yrange)
ax14.set_xlim(xrange)
ax14.set_ylim(yrange)
ax15.set_xlim(xrange)
ax15.set_ylim(yrange)
ax11.set_yscale("log")
ax12.set_yscale("log")
ax13.set_yscale("log")
ax14.set_yscale("log")
ax15.set_yscale("log")

pos_ticks, pos_tick_labels = [-1, 0, 1],  ["-1", "", "1"]
psd_ticks, psd_tick_labels = [0.01, 0.1, 1],  ["0.01", "0.1", "1"]
freq_ticks, freq_tick_labels = [0, 0.5, 1, 1.5, 2],  ["", "$\omega_2$", "$\omega_0$", "$\omega_m$", ""]

ax5.set_xlabel("Time")
ax1.set_ylabel("$\Re[z(t)]$", labelpad=-1, color=clrs[1])
ax2.set_ylabel("$\Re[z_2(t)]$", labelpad=-1, color=clrs[2])
ax3.set_ylabel("$\Re[F_{mask}(t)]$", labelpad=-1, color=clrs[3])
ax4.set_ylabel("$\Re[z(t)]$", labelpad=-1, color=clrs[1])
ax5.set_ylabel("$\Re[z_2(t)]$", labelpad=-1, color=clrs[2])
ax15.set_xlabel("Frequency", labelpad=3)
ax15.set_ylabel("PSD", labelpad=-5)

ax11.set_xticks(freq_ticks, ['', '', '', '', ''])
ax12.set_xticks(freq_ticks, ['', '', '', '', ''])
ax13.set_xticks(freq_ticks, ['', '', '', '', ''])
ax14.set_xticks(freq_ticks, ['', '', '', '', ''])
ax15.set_xticks(freq_ticks, freq_tick_labels)
ax11.set_yticks(psd_ticks, psd_tick_labels)
ax12.set_yticks(psd_ticks, psd_tick_labels)
ax13.set_yticks(psd_ticks, psd_tick_labels)
ax14.set_yticks(psd_ticks, psd_tick_labels)
ax15.set_yticks(psd_ticks, ['', '', '1'])

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])
ax4.set_xticks([])
ax5.set_xticks([])
ax1.set_yticks(pos_ticks, pos_tick_labels)
ax2.set_yticks(pos_ticks, pos_tick_labels)
ax3.set_yticks(pos_ticks, pos_tick_labels)
ax4.set_yticks(pos_ticks, pos_tick_labels)
ax5.set_yticks(pos_ticks, pos_tick_labels)

ax1.spines[['right', 'top']].set_visible(False)
ax2.spines[['right', 'top']].set_visible(False)
ax3.spines[['right', 'top']].set_visible(False)
ax4.spines[['right', 'top']].set_visible(False)
ax5.spines[['right', 'top']].set_visible(False)
ax11.spines[['right', 'top']].set_visible(False)
ax12.spines[['right', 'top']].set_visible(False)
ax13.spines[['right', 'top']].set_visible(False)
ax14.spines[['right', 'top']].set_visible(False)
ax15.spines[['right', 'top']].set_visible(False)


plt.savefig("C:/Users/Justin/Desktop/Fig1.pdf", dpi=300)

'''
plt.figure()
Pxx, freqs, t_bins, im = plt.specgram(np.real(F_mask), NFFT=10*pts_per_cycle, Fs=1/(tt[1]-tt[0]), noverlap=0, scale='linear', 
                                        mode='magnitude', cmap='Greys', vmin=0, vmax=0.004)
plt.ylim(0, 2)
'''

plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after





