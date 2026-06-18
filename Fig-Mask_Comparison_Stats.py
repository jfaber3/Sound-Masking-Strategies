import numpy as np
import matplotlib.pyplot as plt
import code
import time
from scipy.integrate import solve_ivp
from scipy import stats


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

def FM_mask(tot_power, w_mod, a_mod, tt):
    W = 2*np.pi + a_mod*np.sin(w_mod*tt)
    phi = np.cumsum(W*(tt[1]-tt[0]))
    return np.sqrt(tot_power)*np.exp(1j*phi)





num_trials = 0
mask_signal_ratio = 3
seed = 25

#ALL FIXED PARAMETERS
Ff = 0.3 
Fm = 1
mu = 0.1
w0 = 2*np.pi
z0 = [mu**0.5 + 0j]  #initial condition
num_cycles = 1000    #1000
pts_per_cycle = 100  #100
tt = np.linspace(0, num_cycles, num_cycles*pts_per_cycle)
np.random.seed(seed)
mask_power = mask_signal_ratio * (Ff**2)
F_male   = Fm*np.exp(1j*1.5*w0*tt)

def Find_Transfer_Entropy(model, F_female, F_mask):
    if model == 1:
        sol1_m = solve_ivp(make_Hopf_dots_complex(F_female + F_mask), (0, num_cycles), z0, args=(mu, w0), t_eval=tt, method='RK45')
        Z1 = sol1_m.y[0]
        t_ent = Transfer_Entropy(Z1, F_female)
    if model == 2:
        sol_interm_m = solve_ivp(make_Hopf_dots_complex(F_female + F_male + F_mask), (0, num_cycles), z0, args=(mu, w0), t_eval=tt, method='RK45')
        sol2_m = solve_ivp(make_Hopf_dots_complex(sol_interm_m.y[0]), (0, num_cycles), z0, args=(mu, 0.5*w0), t_eval=tt, method='RK45')
        Z2 = sol2_m.y[0]
        t_ent = Transfer_Entropy(Z2, F_female)
    #print(t_ent)
    return t_ent



TE = np.zeros((8, num_trials))
time1 = time.time()
for trial in range(num_trials):
    print(trial)
    F_female = Ff*np.exp(1j*generate_FM_signal(len(tt), tt[1]-tt[0], w0, 0.2*w0))

    #MODEL 1
    TE[0, trial] = Find_Transfer_Entropy(1, F_female, gaussian_noise_mask(mask_power, w0, 0, len(tt), tt[1]-tt[0])     )   #pure tone
    TE[1, trial] = Find_Transfer_Entropy(1, F_female, TwoTone_mask(mask_power, 1.25*w0, 0.25, tt)                      )
    TE[2, trial] = Find_Transfer_Entropy(1, F_female, AM_mask(mask_power, 0.3*w0, 0.07*np.sqrt(mask_signal_ratio), tt) )
    TE[3, trial] = Find_Transfer_Entropy(1, F_female, FM_mask(mask_power, 0.22*w0, 0.08*w0, tt)                        )

    #MODEL 2
    TE[4, trial] = Find_Transfer_Entropy(2, F_female, gaussian_noise_mask(mask_power, 0.975*w0, 0, len(tt), tt[1]-tt[0]) )  #pure tone
    TE[5, trial] = Find_Transfer_Entropy(2, F_female, TwoTone_mask(mask_power, 0.975*w0, 0.25, tt)                       )
    TE[6, trial] = Find_Transfer_Entropy(2, F_female, AM_mask(mask_power, 0.5*w0, 0.1*np.sqrt(mask_signal_ratio), tt)    )
    TE[7, trial] = Find_Transfer_Entropy(2, F_female, FM_mask(mask_power, 0.5*w0, 0.25*w0, tt)                           )

print(np.round( (time.time() - time1)/3600, 4), "hours")



#np.savez(r"C:\Users\Justin\Desktop\TE_Mask_Comparison_Stats.npz", TE=TE)

TE = np.load(r"C:\Users\Justin\Desktop\Auditory Masking\Manuscript Figs v1\TE_Mask_Comparison_Stats.npz")["TE"]


#Doing statistics
t_stat, p_value = stats.ttest_ind(TE[0, :], TE[2, :])
print(f"T-statistic: {t_stat}")
print(f"P-value:     {p_value}")


M1_Ranks = np.argsort(TE[:4, :], axis=0) + 1
M2_Ranks = np.argsort(TE[4:, :], axis=0) + 1
clrs = ["#000000", "#0072B2", "#E69F00", "#009E73"]
labels = ["pure tone", "2 tone", "AM", "FM"]

fig = plt.figure(figsize=(5, 5), constrained_layout=True)
ax1  = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
ax2  = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
ax3  = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
ax4  = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)


num_bins = 10
ax1.hist(TE[0], bins=num_bins, color=clrs[0], histtype='step')
ax1.hist(TE[1], bins=num_bins, color=clrs[1], histtype='step')
ax1.hist(TE[2], bins=num_bins, color=clrs[2], histtype='step')
ax1.hist(TE[3], bins=num_bins, color=clrs[3], histtype='step')
ax2.hist(TE[4], bins=num_bins, color=clrs[0], histtype='step', label=labels[0])
ax2.hist(TE[5], bins=num_bins, color=clrs[1], histtype='step', label=labels[1])
ax2.hist(TE[6], bins=num_bins, color=clrs[2], histtype='step', label=labels[2])
ax2.hist(TE[7], bins=num_bins, color=clrs[3], histtype='step', label=labels[3])

rank_bins = np.array([0.6, 1.4, 1.6, 2.4, 2.6, 3.4, 3.6, 4.4])
ax3.hist(M1_Ranks[0], rank_bins, color=clrs[0], histtype='step')
ax3.hist(M1_Ranks[1], rank_bins, color=clrs[1], histtype='step')
ax3.hist(M1_Ranks[2], rank_bins, color=clrs[2], histtype='step')
ax3.hist(M1_Ranks[3], rank_bins, color=clrs[3], histtype='step')
ax4.hist(M2_Ranks[0], rank_bins, color=clrs[0], histtype='step')
ax4.hist(M2_Ranks[1], rank_bins, color=clrs[1], histtype='step')
ax4.hist(M2_Ranks[2], rank_bins, color=clrs[2], histtype='step')
ax4.hist(M2_Ranks[3], rank_bins, color=clrs[3], histtype='step')


ax1.set_title("Model 1")
ax2.set_title("Model 2")
ax1.set_xlabel("Transfer entropy (bits)")
ax2.set_xlabel("Transfer entropy (bits)")
ax3.set_xlabel("Rank")
ax4.set_xlabel("Rank")
ax1.set_ylabel("Count")
ax2.set_ylabel("Count")
ax3.set_ylabel("Count")
ax4.set_ylabel("Count")
ax3.set_xticks([1, 2, 3, 4])
ax4.set_xticks([1, 2, 3, 4])
ax1.set_ylim(0, 40)
ax2.set_ylim(0, 40)
ax3.set_ylim(0, TE.shape[1])
ax4.set_ylim(0, TE.shape[1])
ax2.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9, 0.1, 0.1), ncols=2, fontsize='small')


plt.savefig("C:/Users/Justin/Desktop/Fig_test.pdf", dpi=300)

plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after











