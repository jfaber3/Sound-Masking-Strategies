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


model = 2
num_trials = 0
MSR = np.linspace(0, 5, 26)


TE = np.zeros(len(MSR))
time1 = time.time()
for trial in range(num_trials):
    print(trial)
    F_female = Ff*np.exp(1j*generate_FM_signal(len(tt), tt[1]-tt[0], w0, 0.2*w0))
    for i in range(len(MSR)):
        mask_power = MSR[i] * Ff**2
        #F_mask   = gaussian_noise_mask(mask_power, w0, np.inf, len(tt), tt[1]-tt[0])   #white noise (M1 and M2)
        #F_mask   = gaussian_noise_mask(mask_power, w0, 0.3*w0, len(tt), tt[1]-tt[0])   #bandpass filtered noise (M1 and M2)
        
        #MODEL 1
        #F_mask   = gaussian_noise_mask(mask_power, w0,       0, len(tt), tt[1]-tt[0])     #pure tone M1
        #F_mask    = TwoTone_mask(mask_power, 1.25*w0, 0.25, tt)                           #two  tone M1
        #F_mask    = AM_mask(mask_power, 0.3*w0, 0.07*np.sqrt(MSR[i]), tt) #scale with msr #AM        M1
        #F_mask    = FM_mask(mask_power, 0.22*w0, 0.08*w0, tt)       #good,  broad         #FM        M1
        #OTHER OPTIONS
        #F_mask    = FM_power_mask(mask_power, 0.02*w0, 0.5, tt)    #good
        #F_mask    = FM_square_mask(mask_power, 0.14, 0.05*w0, tt)  #good
        #F_mask    = FM_linear_mask(mask_power, 0.5 , 0.1*w0,  tt)   #less good
        
        #MODEL 2
        #F_mask    = gaussian_noise_mask(mask_power, 0.975*w0, 0, len(tt), tt[1]-tt[0])        #pure tone M2
        #F_mask    = TwoTone_mask(mask_power, 0.975*w0, 0.25, tt)                             #two  tone M2
        #F_mask    = AM_mask(mask_power, 0.5*w0, 0.1*np.sqrt(MSR[i]), tt) #scale with msr     #AM        M2
        F_mask    = FM_mask(mask_power, 0.5*w0, 0.25*w0, tt)         #good,  broad           #FM        M2



        if model == 1:
            sol1_m = solve_ivp(make_Hopf_dots_complex(F_female + F_mask), (0, num_cycles), z0, args=(mu, w0), t_eval=tt, method='RK45')
            Z1 = sol1_m.y[0]
            TE[i] += Transfer_Entropy(Z1, F_female)/num_trials
            
        if model == 2:
            sol_interm_m = solve_ivp(make_Hopf_dots_complex(F_female + F_male + F_mask), (0, num_cycles), z0, args=(mu, w0), t_eval=tt, method='RK45')
            sol2_m = solve_ivp(make_Hopf_dots_complex(sol_interm_m.y[0]), (0, num_cycles), z0, args=(mu, wdp), t_eval=tt, method='RK45')
            Z2 = sol2_m.y[0]
            TE[i] += Transfer_Entropy(Z2, F_female)/num_trials

if num_trials > 0:
    print(np.round( (time.time() - time1)/3600, 4), "hours")
    print("Transfer entropy")
    print(np.array2string(np.round(TE, 5), separator=', ')[1:-1])
    plt.figure()
    plt.plot(MSR, TE, "-", color='black')
    plt.xlabel("Mask-to-signal ratio")
    plt.ylabel("Transfer entropy (bits)")
    plt.ylim(0, 1.1*max(TE))
    plt.figure()
    plt.plot(np.real(F_mask))
    plt.plot(np.imag(F_mask))
    plt.figure()
    plt.plot(np.real(F_mask), np.imag(F_mask))





#MODEL 1, white noise  (50 trials)
TE1_1 = np.array([0.14461, 0.14432, 0.1436 , 0.14338, 0.14266, 0.14224, 0.14191, 0.14129,
 0.1416 , 0.14063, 0.14063, 0.14045, 0.13998, 0.13954, 0.1388 , 0.13866,
 0.13894, 0.13857, 0.13774, 0.1378 , 0.13766, 0.13703, 0.13699, 0.13701,
 0.13655, 0.13575])

#MODEL 1, filtered noise  wdev=0.3w0 (300 trials)
TE1_2 = np.array([0.14457, 0.12975, 0.11718, 0.10609, 0.09669, 0.08893, 0.08187, 0.07697,
 0.07215, 0.06781, 0.06441, 0.06165, 0.05848, 0.05616, 0.05361, 0.05217,
 0.05033, 0.04859, 0.04681, 0.04637, 0.04513, 0.04384, 0.04266, 0.04249,
 0.04056, 0.04038])

#MODEL 1, pure tone  w = w0  (50 trials)
TE1_3 = np.array([0.14461, 0.10294, 0.0732 , 0.05334, 0.04158, 0.03656, 0.03279, 0.03059,
 0.02836, 0.02693, 0.02595, 0.02502, 0.0244 , 0.02289, 0.02245, 0.02167,
 0.02127, 0.02086, 0.02055, 0.02016, 0.01958, 0.01947, 0.01874, 0.01906,
 0.01815, 0.01818])

#MODEL 1, Two tone, amp_ratio = 0.25. w1 = 1.01w0, w2= 1.25w0   (50 trials)
TE1_4 = np.array([0.14527, 0.10533, 0.07668, 0.05735, 0.04373, 0.03594, 0.03167, 0.02873,
 0.02631, 0.02436, 0.02278, 0.02143, 0.0203 , 0.0192 , 0.01839, 0.01749,
 0.01681, 0.0162 , 0.01566, 0.01515, 0.01467, 0.01421, 0.01377, 0.01341,
 0.01314, 0.01283])

#MODEL 1, AM, w_mod = 0.3w0, A_mod = 0.07 (scaled by root(MSR))   (50 trials)
TE1_5 = np.array([0.14527, 0.10468, 0.07544, 0.05609, 0.04333, 0.0361 , 0.03228, 0.02964,
 0.02748, 0.02563, 0.02415, 0.02295, 0.02203, 0.02119, 0.02037, 0.01966,
 0.01905, 0.01844, 0.01796, 0.01748, 0.01702, 0.01658, 0.01621, 0.01592,
 0.0156 , 0.01531])

#MODEL 1, FM, sin mod, w_mod = 0.22*w0, A_mod = 0.08*w0  (50 trials)
TE1_6 = np.array([0.14527, 0.10376, 0.07511, 0.05601, 0.04264, 0.03586, 0.03132, 0.02739,
 0.02505, 0.02345, 0.0221 , 0.02052, 0.01889, 0.01737, 0.01632, 0.01554,
 0.01503, 0.01465, 0.0143 , 0.01394, 0.01361, 0.01329, 0.01297, 0.01266,
 0.01235, 0.01206])






#MODEL 2, white noise  (100 trials)
TE2_1 = np.array([0.16012, 0.15972, 0.15963, 0.15895, 0.15897, 0.15858, 0.1581 , 0.15816,
 0.15801, 0.15801, 0.157  , 0.15671, 0.15704, 0.15686, 0.15699, 0.15605,
 0.15631, 0.15594, 0.15561, 0.15545, 0.15529, 0.15535, 0.15471, 0.15454,
 0.15463, 0.15443])

#MODEL 2, filtered noise   wdev=0.3w0 (100 trials)
TE2_2 = np.array([0.16012, 0.15145, 0.14218, 0.13397, 0.12683, 0.12034, 0.11362, 0.10835,
 0.10274, 0.09849, 0.09417, 0.09059, 0.08735, 0.08469, 0.08032, 0.07875,
 0.07624, 0.07353, 0.07118, 0.06991, 0.06766, 0.06638, 0.06358, 0.06266,
 0.0613 , 0.0599])

#MODEL 2, pure tone,  w = 0.975*w0 (300 trials)
TE2_3 = np.array([0.16025, 0.13208, 0.10863, 0.09044, 0.07866, 0.07053, 0.06568, 0.06229,
 0.0609 , 0.06005, 0.05926, 0.05884, 0.05892, 0.05867, 0.05902, 0.05892,
 0.05946, 0.05966, 0.05988, 0.05959, 0.06024, 0.05935, 0.05885, 0.05846,
 0.05803, 0.05777])

#MODEL 2, two tone, w2 = 0.975*w0, w1 = 1.01w0, amp_ratio = 0.25  (100 trials)
TE2_4 = np.array([0.15952, 0.12953, 0.10506, 0.08696, 0.07488, 0.06689, 0.06157, 0.05788,
 0.05515, 0.05312, 0.05159, 0.05031, 0.04927, 0.04833, 0.04765, 0.04703,
 0.04648, 0.04595, 0.04553, 0.0453 , 0.04498, 0.04482, 0.04449, 0.04419,
 0.04394, 0.04374])

#MODEL 2, AM, w_mod = wdp, A_mod = 0.1 (scaled by root(MSR))   (0 trials)
TE2_5 = np.array([0.15952, 0.13021, 0.10568, 0.08799, 0.07658, 0.06878, 0.06347, 0.05919,
 0.05602, 0.05349, 0.05129, 0.04937, 0.04786, 0.04645, 0.0453 , 0.04423,
 0.04327, 0.04238, 0.04164, 0.04092, 0.04033, 0.03976, 0.0392 , 0.03871,
 0.03831, 0.0379])

#MODEL 2, FM sin mod, w_mod = wdp, A_mod = 0.25   (0 trials)
TE2_6 = np.array([0.15952, 0.13192, 0.11057, 0.09281, 0.07815, 0.06755, 0.06023, 0.05533,
 0.05152, 0.04854, 0.04604, 0.04398, 0.04237, 0.04099, 0.03986, 0.03883,
 0.03799, 0.03723, 0.03664, 0.03614, 0.03572, 0.03526, 0.03491, 0.0346 ,
 0.03429, 0.03401])







clrs = ["#000000", "#0072B2", "#E69F00", "#009E73", "#9E0084"]
legend_labels1 = ["white noise", "filtered noise", "pure tone", "2 tone", "AM", "FM"]
legend_labels2 = ["white noise", "filtered noise", "pure tone", "2 tone", "AM", "FM"]

fig = plt.figure(figsize=(5, 7))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.07, top=0.88, hspace=0.15)
ax1  = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
ax2  = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)
ax1.set_ylabel("Transfer entropy (bits)")
ax2.set_ylabel("Transfer entropy (bits)")


ax1.plot(MSR, TE1_1, ":",  color=clrs[0], label=legend_labels1[0])
ax1.plot(MSR, TE1_2, "--", color=clrs[0], label=legend_labels1[1])
ax1.plot(MSR, TE1_3,       color=clrs[0], label=legend_labels1[2])
ax1.plot(MSR, TE1_4,       color=clrs[1], label=legend_labels1[3])
ax1.plot(MSR, TE1_5,       color=clrs[2], label=legend_labels1[4])
ax1.plot(MSR, TE1_6,       color=clrs[3], label=legend_labels1[5])

ax2.plot(MSR, TE2_1, ":",  color=clrs[0], label=legend_labels2[0])
ax2.plot(MSR, TE2_2, "--", color=clrs[0], label=legend_labels2[1])
ax2.plot(MSR, TE2_3,       color=clrs[0], label=legend_labels2[2])
ax2.plot(MSR, TE2_4,       color=clrs[1], label=legend_labels2[3])
ax2.plot(MSR, TE2_5,       color=clrs[2], label=legend_labels2[4])
ax2.plot(MSR, TE2_6,       color=clrs[3], label=legend_labels2[5])

ax1.set_xlim(0, 5)
ax2.set_xlim(0, 5)
ax1.set_ylim(0, 0.17)
ax2.set_ylim(0, 0.17)
ax2.set_xlabel("Mask-to-signal ratio")
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.32), ncols=2)
#ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.text(-0.12, 1.08, "A", transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
ax2.text(-0.12, 1.08, "B", transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
plt.savefig("C:/Users/Justin/Desktop/Fig_test.pdf", dpi=300)

plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after

















'''
MSR = np.logspace(-3, 3, 25)

#MODEL 1, white noise  (50 trials)
TE1_1 = np.array([0.14387, 0.14386, 0.14389, 0.14368, 0.14361, 0.14362, 0.14359, 0.14363,
 0.14362, 0.1434 , 0.14339, 0.14272, 0.14175, 0.14025, 0.1389 , 0.1357 ,
 0.12884, 0.11951, 0.10691, 0.09138, 0.0737 , 0.06029, 0.04937, 0.03953,
 0.03467])

#MODEL 1, filtered noise  wdev=0.3w0 (50 trials)
TE1_2 = np.array([0.14395, 0.14379, 0.14387, 0.14379, 0.14339, 0.14283, 0.14175, 0.13997,
 0.13758, 0.13153, 0.12224, 0.10912, 0.08887, 0.06901, 0.05141, 0.03853,
 0.02955, 0.02451, 0.02194, 0.02125, 0.02067, 0.02133, 0.02181, 0.02217,
 0.02265])

#MODEL 1, pure tone  w = w0  (50 trials)
TE1_3 = np.array([0.14369, 0.14359, 0.14332, 0.14312, 0.14176, 0.14024, 0.13669, 0.13148,
 0.1214 , 0.10546, 0.08273, 0.05586, 0.03607, 0.02717, 0.02162, 0.01794,
 0.016  , 0.01758, 0.02095, 0.01742, 0.01528, 0.00913, 0.00806, 0.00825,
 0.00663])

#MODEL 1, Two tone, amp_ratio = 0.25. w1 = 1.01w0, w2= 1.25w0   (50 trials)
TE1_4 = np.array([0.14512, 0.14482, 0.14442, 0.14395, 0.14291, 0.14119, 0.13784, 0.13236,
 0.12324, 0.1091 , 0.08766, 0.06055, 0.03594, 0.02456, 0.01696, 0.01191,
 0.0086 , 0.00647, 0.0051 , 0.00437, 0.00388, 0.00361, 0.00343, 0.00335,
 0.00327])

#MODEL 1, AM, w_mod = 0.3w0, A_mod = 0.07 (scaled by root(MSR))   (50 trials)
TE1_5 = np.array([0.14513, 0.14482, 0.14438, 0.14397, 0.14289, 0.14104, 0.13777, 0.13205,
 0.12289, 0.10845, 0.08626, 0.05926, 0.0361 , 0.02583, 0.01916, 0.01459,
 0.01126, 0.00913, 0.00734, 0.00595, 0.00465, 0.00359, 0.00266, 0.00214,
 0.0019])

#MODEL 1, FM, sin mod, w_mod = 0.22*w0, A_mod = 0.08*w0  (50 trials)
TE1_6 = np.array([0.14504, 0.14476, 0.14423, 0.14375, 0.14242, 0.1403 , 0.13719, 0.13149,
 0.12205, 0.10751, 0.08598, 0.05938, 0.03586, 0.02364, 0.01511, 0.01098,
 0.00657, 0.00601, 0.0066 , 0.00548, 0.00387, 0.00354, 0.00348, 0.00349,
 0.00349])




#MODEL 2, white noise  (50 trials)
TE2_1 = np.array([0.15939, 0.15945, 0.15955, 0.15965, 0.15965, 0.15971, 0.15966, 0.15919,
 0.15915, 0.1591 , 0.15881, 0.15824, 0.15849, 0.1569 , 0.15582, 0.1532 ,
 0.1494 , 0.14321, 0.13156, 0.11809, 0.0966 , 0.07235, 0.05558, 0.04048,
 0.03236])

#MODEL 2, filtered noise   wdev=0.3w0 (50 trials)
TE2_2 = np.array([0.15928, 0.15954, 0.15911, 0.15928, 0.15937, 0.15903, 0.15837, 0.15648,
 0.15452, 0.1515 , 0.1451 , 0.1352 , 0.12071, 0.09958, 0.07734, 0.05769,
 0.04262, 0.03395, 0.02799, 0.02555, 0.02348, 0.0228 , 0.02271, 0.02213,
 0.02229])

#MODEL 2, pure tone,  w = 0.975*w0
TE2_3 = np.array([0.15934, 0.15944, 0.15932, 0.15877, 0.15864, 0.15638, 0.15473, 0.15121,
 0.14509, 0.1351 , 0.11787, 0.09336, 0.0711 , 0.05914, 0.05885, 0.05516,
 0.0352 , 0.01877, 0.01108, 0.00672, 0.00504, 0.00428, 0.00366, 0.00372,
 0.00204])

#MODEL 2, two tone, w2 = 0.975*w0, w1 = 1.01w0, amp_ratio = 0.25
TE2_4 = np.array([0.15954, 0.15952, 0.15925, 0.15897, 0.15848, 0.15693, 0.1549 , 0.15081,
 0.14397, 0.13265, 0.11416, 0.08921, 0.06608, 0.05303, 0.04665, 0.04367,
 0.04122, 0.0347 , 0.02313, 0.01491, 0.01023, 0.00636, 0.0045 , 0.0037 ,
 0.00346])

#MODEL 2, AM, w_mod = wdp, A_mod = 0.1 (scaled by root(MSR))   (50 trials)
TE2_5 = np.array([0.1596 , 0.15952, 0.15924, 0.15895, 0.1584 , 0.15728, 0.15525, 0.15149,
 0.14464, 0.13307, 0.1148 , 0.09071, 0.06885, 0.05342, 0.0431 , 0.03658,
 0.03294, 0.03111, 0.03112, 0.02911, 0.01722, 0.01208, 0.00723, 0.00601,
 0.00467])

#MODEL 2, FM sin mod, w_mod = wdp, A_mod = 0.25   (1 trials)
TE2_6 = np.array([0.1594 , 0.15923, 0.15906, 0.15839, 0.15766, 0.15595, 0.15401, 0.15003,
 0.14378, 0.13422, 0.11888, 0.09576, 0.06722, 0.04811, 0.03749, 0.03265,
 0.03014, 0.02034, 0.01336, 0.01311, 0.03702, 0.03276, 0.01796, 0.02904,
 0.0153])

'''






'''
#MODEL 2 pure tone, 400 trials
TE2_3 = np.array([0.16002, 0.15989, 0.15975, 0.15923, 0.15876, 0.15742, 0.15524, 0.15145,
 0.14543, 0.13444, 0.11757, 0.09314, 0.0709 , 0.05956, 0.05927, 0.05488,
 0.03503, 0.01869, 0.01096, 0.00683, 0.00504, 0.00422, 0.00369, 0.00364,
 0.00211])
'''




