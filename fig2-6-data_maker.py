import numpy as np
import matplotlib.pyplot as plt
import code
import time
from scipy.integrate import solve_ivp



#Transfer Entropy Functions

def make_digital_boxes(x):  # 0 or 1 based on below or above mean
    x_boxes = np.zeros(len(x), dtype=int)
    x_avg = np.mean(x)
    for i in range(len(x)):
        if x[i] > x_avg:
            x_boxes[i] = 1
    return x_boxes

def make_states(boxes, dim, tau):  # max 9 dimensions
    states = []
    for i in range(len(boxes) - dim*tau + 1):
        word = ''
        for j in range(dim):
            word += str(boxes[i + tau*j])
        states.append(word)
    return states

def indexer(x_states):  # takes states like '012302' and renames them integers 0, 1, 2, 3...
    new_states = []
    sorted_states = np.sort(x_states)
    unique_states = []
    unique_states.append(sorted_states[0])
    for i in range(1, len(sorted_states)):
        if sorted_states[i] != sorted_states[i - 1]:
            unique_states.append(sorted_states[i])
    num_unique_states = len(unique_states)
    for i in range(len(x_states)):
        for j in range(num_unique_states):
            if x_states[i] == unique_states[j]:
                new_states.append(j)
                break
    return new_states

def T_Entropy(x_states_plus1, y_states):  # if dim=5, this uses first 4 dims to find probability of getting the 5th.  infromation transfered from y to x
    y_states = y_states[:len(x_states_plus1)]  # cut off last few to make len x and y states match
    next_x = []
    x_states = []
    for i in range(len(x_states_plus1)):
        word = x_states_plus1[i]
        next_x.append(word[-1])
        x_states.append(word[:-1])

    new_x_states = indexer(x_states)  # re naming states to integers 0, 1, 2, 3...
    new_y_states = indexer(y_states)
    new_x_next_states = indexer(next_x)

    num_x_states = max(new_x_states) + 1
    num_y_states = max(new_y_states) + 1
    num_x_next_states = max(new_x_next_states) + 1
    # print num_x_states, num_y_states, num_x_next_states
    assert len(new_x_states) == len(new_y_states)

    Big_P = np.zeros((num_x_states, num_y_states, num_x_next_states), dtype=float)
    P_x_next_condit_X = np.zeros((num_x_states, num_x_next_states), dtype=float)
    P_x_next_condit_XY = np.zeros((num_x_states, num_y_states, num_x_next_states), dtype=float)

    for i in range(len(new_x_states)):
        Big_P[new_x_states[i]][new_y_states[i]][new_x_next_states[i]] += 1.0
        P_x_next_condit_X[new_x_states[i]][new_x_next_states[i]] += 1.0
        P_x_next_condit_XY[new_x_states[i]][new_y_states[i]][new_x_next_states[i]] += 1.0

    Big_P /= np.sum(Big_P)  # Normalizing
    for i in range(num_x_states):
        if np.sum(P_x_next_condit_X[i]) > 0.0:
            P_x_next_condit_X[i] /= np.sum(P_x_next_condit_X[i])  # Normalizing based on conditional info
    for i in range(num_x_states):
        for j in range(num_y_states):
            if np.sum(P_x_next_condit_XY[i][j]) > 0.0:
                P_x_next_condit_XY[i][j] /= np.sum(P_x_next_condit_XY[i][j])  # Normalizing based on conditional info
    T_entropy = 0.0
    for i in range(num_x_states):
        for j in range(num_y_states):
            for k in range(num_x_next_states):
                if P_x_next_condit_XY[i][j][k]*P_x_next_condit_X[i][k] > 0.0:
                    T_entropy += Big_P[i][j][k]*( np.log2(P_x_next_condit_XY[i][j][k]) - np.log2(P_x_next_condit_X[i][k]) )
    return T_entropy  # Big_P, P_x_next_condit_X, P_x_next_condit_XY, new_x_states, new_y_states, new_x_next_states

def calc_TE(x, y, dim, tau):  # info tranfered from y to x
    x_boxes = make_digital_boxes(x)
    y_boxes = make_digital_boxes(y)
    x_states_plus1 = make_states(x_boxes, dim + 1, tau)  # 1 extra dimension to be used in the prediction
    y_states = make_states(y_boxes, dim, tau)
    return T_Entropy(x_states_plus1, y_states)

def Transfer_Entropy(Response, Signal, subsamp=5, point_per_cycle=100):  #complex signal and response
    Res_x, Res_y = np.real( Response[::subsamp] ), np.imag( Response[::subsamp] )
    Sig_x, Sig_y = np.real(   Signal[::subsamp] ), np.imag(   Signal[::subsamp] )
    dim = 5
    tau = int(point_per_cycle/(dim*subsamp))
    te_x = calc_TE(Res_x, Sig_x, dim, tau)
    te_y = calc_TE(Res_y, Sig_y, dim, tau)
    return (te_x + te_y)/2




#Signal Processing Functions

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


#MASK FUNCTIONS

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

def TwoTone_mask(tot_power, w2_mask, amp_ratio, tt):
    w1_mask = 1.01*(2*np.pi)
    A1 = np.sqrt(tot_power/(1 + amp_ratio**2))
    A2 = amp_ratio*A1
    return A1*np.exp(1j*w1_mask*tt) + A2*np.exp(1j*w2_mask*tt)

def AM_mask(tot_power, w_mod, a_mod, tt):
    w_mask = 1.01*(2*np.pi)
    assert a_mod <= np.sqrt(2*tot_power)
    a0 = np.sqrt(tot_power - 0.5*a_mod**2)
    return (a0 + a_mod*np.sin(w_mod*tt))*np.exp(1j*w_mask*tt)



def FM_mask(tot_power, w_mod, a_mod, tt):                              #CORRECTION - USE 1.01 w0?
    W = 2*np.pi + a_mod*np.sin(w_mod*tt)
    phi = np.cumsum(W*(tt[1]-tt[0]))
    return np.sqrt(tot_power)*np.exp(1j*phi)

def FM_linear_mask(tot_power, f_mod, a_mod, tt):  #linear ramp up      #CORRECTION - USE 1.01 w0?
    W = 2*np.pi + a_mod*( 2*(  (f_mod*tt)%(1)  ) - 1)  
    phi = np.cumsum(W*(tt[1]-tt[0]))
    return np.sqrt(tot_power)*np.exp(1j*phi)

def FM_power_mask(tot_power, a_mod, power, tt):  #power law ramp up    #CORRECTION - USE 1.01 w0?
    f_mod = 0.02  #fixed modulation frequency
    x = 2*( (f_mod*tt)%(1) ) - 1  # runs from -1 to 1
    W = 2*np.pi + a_mod*x*abs(x)**(power - 1)
    phi = np.cumsum(W*(tt[1]-tt[0]))
    return np.sqrt(tot_power)*np.exp(1j*phi)

def FM_square_mask(tot_power, f_mod, a_mod, tt):  #square wave modulator: discret jumps between 2 frequencies
    W = 2*np.pi + a_mod*np.sign( 2*((f_mod*tt)%(1)) - 1 )  
    phi = np.cumsum(W*(tt[1]-tt[0]))
    return np.sqrt(tot_power)*np.exp(1j*phi)



#ALL FIXED PARAMETERS
Ff = 0.3 
Fm = 1
mask_power = 1 * Ff**2
mu = 0.1
w0 = 2*np.pi
wdp =  np.pi
z0 = [mu**0.5 + 0j]  #initial condition
num_cycles = 1000    #1000
pts_per_cycle = 100  #100
tt = np.linspace(0, num_cycles, num_cycles*pts_per_cycle)
F_male   = Fm*np.exp(1j*1.5*w0*tt)
seed = 12  #always 12
np.random.seed(seed)
num_trials = 50    #number of female stimulus waveforms (at least 60)


'''
mask_type = "noise_mask_p1"
fixed_param = np.array([ 0, 0.1*w0, 0.2*w0])    #panal 1 - sdev of noise mask
sweep_param = np.linspace(0.2*w0, 1.8*w0, 65)    #(65) panal 1 - center frequency 
'''
'''
mask_type = "noise_mask_p2"
fixed_param = np.array([ w0, 1.1*w0, 0.5*w0])    #panal 2 - center frequency
sweep_param = np.linspace(0, w0, 40)    #(65) panal 2 - sdev of noise mask
'''
'''
mask_type = "TwoTone_mask_p1"
fixed_param = np.array([ 0.1, 0.5,  1])    #panal 1 - amp ratio of 2tone mask 
sweep_param = np.linspace(0.2*w0, 1.8*w0, 65)    #frequency of second tone, 60pts
'''
'''
mask_type = "TwoTone_mask_p2"
fixed_param = np.array([ 0.975*w0, 1.25*w0, 0.5*w0 ])    #panal 2 frequency of second tone
sweep_param = np.linspace(0, 1, 40)    #amp ratio of 2tone mask 
'''
'''
mask_type = "AM_mask_p1"
fixed_param = np.array([ 0.1, 0.2, np.sqrt(2*Ff**2) ])    #panal 1 - amplitude of modulation (a_mod) (max value = root(2*tot_power))
sweep_param = np.linspace(0, w0, 41)                       #modulator frequency (w_mod)
'''
'''
mask_type = "AM_mask_p2"
fixed_param = np.array([ 0.05*w0, 0.3*w0, 0.5*w0 ])        #panal 2 - modulator frequency (w_mod)
sweep_param = np.linspace(0, np.sqrt(2*Ff**2), 30)                       # amplitude of modulation (a_mod) (max value = root(2*tot_power))
'''
'''
mask_type = "FM_mask_p1"
fixed_param = np.array([ 0.05*w0, 0.1*w0, 0.2*w0 ])              #panal 1 - magnitude of modulation (a_mod)
sweep_param = np.linspace(0.02*w0, w0, 50)                       # modulator frequency (w_mod)
'''
'''
mask_type = "FM_mask_p2"
fixed_param = np.array([ 0.05*w0, 0.2*w0, 0.5*w0 ])              #panal 2 - modulator frequency (w_mod)
sweep_param = np.linspace(0, 0.5*w0, 51)                         #magnitude of modulation (a_mod)
'''
'''
mask_type = "FM_linear_mask_p1"  #(0.05 and 0.3 good, keep 0.1),    original: 0.05*w0, 0.1*w0, 0.2*w0
fixed_param = np.array([ 0.04*w0, 0.07*w0, 0.34*w0 ])         #panal 1 - magnitude of modulation (a_mod)  mins at 0.04, 0.07, 0.34
sweep_param = np.linspace(0.02, 1, 50)                      # modulator frequency (f_mod)  
'''
'''
mask_type = "FM_linear_mask_p2"
fixed_param = np.array([ 0.02, 0.1, 0.5 ])                    # panal 2 - modulator frequency (f_mod)
sweep_param = np.linspace(0, 0.5*w0, 51)                           #magnitude of modulation (a_mod)
'''
'''
mask_type = "FM_power_mask_p1"
fixed_param = np.array([ 0.5, 1, 2 ])                            # panal 1 - power of growth freq^power
sweep_param = np.linspace(0, 0.2*w0, 41)                         # magnitude of modulation (a_mod)   np.linspace(0, 0.2*w0, 41)
'''
'''
mask_type = "FM_power_mask_p2"
fixed_param = np.array([ 0.02*w0, 0.04*w0, 0.06*w0 ])          # panal 2 - magnitude of modulation (a_mod)
sweep_param = np.linspace(0, 4, 41)                            # power of growth freq^power 
'''
'''
mask_type = "FM_square_mask_p1"
fixed_param = np.array([ 0.05*w0, 0.1*w0, 0.2*w0 ])                     # panal 1  - magnitude of modulation (a_mod)
sweep_param = np.linspace(0.02, 1, 50)                                              # modulator frequency (f_mod) 
'''

mask_type = "FM_square_mask_p2"
fixed_param = np.array([ 0.14, 0.48, 0.8 ])                     # panal 2  - # modulator frequency (f_mod)
sweep_param = np.linspace(0, 0.5*w0, 51)                                     # magnitude of modulation (a_mod)





time1 = time.time()
TE1, TE2 = np.zeros(( len(fixed_param), len(sweep_param) )), np.zeros(( len(fixed_param), len(sweep_param) ))
for trial in range(num_trials):
    print("trial = ", trial)
    F_female = Ff*np.exp(1j*generate_FM_signal(len(tt), tt[1]-tt[0], w0, 0.2*w0))
    for j in range(len(fixed_param)):
        print(j)
        for i in range(len(sweep_param)):
            #F_mask   = gaussian_noise_mask(mask_power, sweep_param[i], fixed_param[j], len(tt), tt[1]-tt[0]) 
            #F_mask   = gaussian_noise_mask(mask_power, fixed_param[j], sweep_param[i], len(tt), tt[1]-tt[0]) 
            #F_mask   = TwoTone_mask(mask_power, sweep_param[i], fixed_param[j], tt)
            #F_mask   = TwoTone_mask(mask_power, fixed_param[j], sweep_param[i], tt)
            
            #F_mask   = AM_mask(mask_power, sweep_param[i], fixed_param[j], tt)
            #F_mask   = AM_mask(mask_power, fixed_param[j], sweep_param[i], tt)
            #F_mask   = FM_mask(mask_power, sweep_param[i], fixed_param[j], tt)
            #F_mask   = FM_mask(mask_power, fixed_param[j], sweep_param[i], tt)
            
            #F_mask   = FM_linear_mask(mask_power, sweep_param[i], fixed_param[j], tt)
            #F_mask   = FM_linear_mask(mask_power, fixed_param[j], sweep_param[i], tt)
            #F_mask   = FM_power_mask(mask_power, sweep_param[i], fixed_param[j], tt)
            #F_mask   = FM_power_mask(mask_power, fixed_param[j], sweep_param[i], tt)
            #F_mask   = FM_square_mask(mask_power, sweep_param[i], fixed_param[j], tt)
            F_mask   = FM_square_mask(mask_power, fixed_param[j], sweep_param[i], tt)
            
            
            
            #1 Hopf with mask
            sol1_m = solve_ivp(make_Hopf_dots_complex(F_female + F_mask), (0, num_cycles), z0, args=(mu, w0), t_eval=tt, method='RK45')
            Z1 = sol1_m.y[0]
            #2 Hopf with mask
            sol_interm_m = solve_ivp(make_Hopf_dots_complex(F_female + F_male + F_mask), (0, num_cycles), z0, args=(mu, w0), t_eval=tt, method='RK45')
            sol2_m = solve_ivp(make_Hopf_dots_complex(sol_interm_m.y[0]), (0, num_cycles), z0, args=(mu, wdp), t_eval=tt, method='RK45')
            Z2 = sol2_m.y[0]
            TE1[j, i] += Transfer_Entropy(Z1, F_female)/num_trials
            TE2[j, i] += Transfer_Entropy(Z2, F_female)/num_trials


np.save("C:/Users/Justin/Desktop/TE_Data_" + mask_type + "_seed=" + str(seed) + "_trials=" + str(num_trials), 
        np.array([fixed_param, sweep_param, TE1, TE2], dtype=object) )

print(np.round( (time.time() - time1)/3600, 4), "hours")
#Loading Data
#fixed_param, sweep_param, TE1, TE2 = np.load("C:/Users/Justin/Desktop/TE_Data_noise_mask_1_seed=12_trials=3.npy", allow_pickle=True)

line_styles = ['-', '--', ':']
plt.figure()
for j in range(len(fixed_param)):
    plt.plot(sweep_param/(2*np.pi), TE1[j, :], line_styles[j], color='blue')   #/(2*np.pi)
    plt.plot(sweep_param/(2*np.pi), TE2[j, :], line_styles[j], color='orange')
#plt.ylim(0.0, 0.18)


plt.figure()   #SPECTROGRAM
Pxx, freqs, t_bins, im = plt.specgram(np.real(F_mask), NFFT=10*pts_per_cycle, Fs=1/(tt[1]-tt[0]), noverlap=0, scale='linear', 
                                        mode='magnitude', cmap='Greys', vmin=0, vmax=0.004)
plt.ylim(0, 2)



plt.show()


code.interact(local=locals())  #allows interaction with variables in terminal after







'''

num_seg = 20
f_male, xf_male     = PSD_avg_segs(np.real(F_male), pts_per_cycle, num_seg)
f_female, xf_female = PSD_avg_segs(np.real(F_female), pts_per_cycle, num_seg)
f_mask, xf_mask     = PSD_avg_segs(np.real(F_mask), pts_per_cycle,   num_seg)
f1,  xf1,              = PSD_avg_segs(x1,  pts_per_cycle, num_seg)
f1m, xf1m,             = PSD_avg_segs(x1m, pts_per_cycle, num_seg)
f2,  xf2,              = PSD_avg_segs(x2,  pts_per_cycle, num_seg)
f2m, xf2m,             = PSD_avg_segs(x2m, pts_per_cycle, num_seg)



fig = plt.figure(figsize=(5, 5))
plt.subplots_adjust(left=0.08, right=0.97, bottom=0.1, top=0.99, wspace=0.5, hspace=0.1)
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


xrange = (num_cycles-30, num_cycles-10)
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
yrange = (0.001, 3)
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

pos_ticks, pos_tick_labels = [-1, 0, 1],  ["-1", "0", "1"]
psd_ticks, psd_tick_labels = [0.01, 0.1, 1],  ["0.01", "0.1", "1"]
freq_ticks, freq_tick_labels = [0, 0.5, 1, 1.5, 2],  ["", "$\omega_{dp}$", "$\omega_0$", "$\omega_m$", ""]

ax5.set_xlabel("Time")
ax5.set_ylabel("Response", labelpad=3)
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
ax5.set_yticks(pos_ticks, ['', '', ''])

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


#plt.savefig("C:/Users/Justin/Desktop/Fig2.pdf", dpi=300)

plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after


'''


