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


def AM_mask(tot_power, w_carrier, w_mod, a_mod, tt):
    w_mask = w_carrier
    assert a_mod <= np.sqrt(2*tot_power)
    a0 = np.sqrt(tot_power - 0.5*a_mod**2)
    return (a0 + a_mod*np.sin(w_mod*tt))*np.exp(1j*w_mask*tt)

def FM_mask(tot_power, w_carrier, w_mod, a_mod, tt):          
    W = w_carrier + a_mod*np.sin(w_mod*tt)
    phi = np.cumsum(W*(tt[1]-tt[0]))
    return np.sqrt(tot_power)*np.exp(1j*phi)



def FM_linear_mask(tot_power, f_mod, a_mod, tt):  #linear ramp up    
    W = 2*np.pi + a_mod*( 2*(  (f_mod*tt)%(1)  ) - 1)  
    phi = np.cumsum(W*(tt[1]-tt[0]))
    return np.sqrt(tot_power)*np.exp(1j*phi)

def FM_power_mask(tot_power, a_mod, power, tt):  #power law ramp up   
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
z0 = [mu**0.5 + 0j]  #initial condition
num_cycles = 1000    #1000
pts_per_cycle = 100  #100
tt = np.linspace(0, num_cycles, num_cycles*pts_per_cycle)
seed = 12  #always 12
np.random.seed(seed)
mask_power = Ff**2




num_trials = 0  #50
Carrier = np.linspace(0.2, 1.8, 31)*2*np.pi


TE = np.zeros((len(Carrier), num_trials))
time1 = time.time()
for trial in range(num_trials):
    print(trial)
    for i in range(len(Carrier)):
        w_carrier = Carrier[i]
        F_female = Ff*np.exp(1j*generate_FM_signal(len(tt), tt[1]-tt[0], w0, 0.2*w0))
        F_male   = Fm*np.exp(1j*1.5*w0*tt)
        
        #MODEL 1
        #F_mask, model    = AM_mask(mask_power, w_carrier, 0.3*w0, 0.07, tt),  1    
        F_mask, model    = FM_mask(mask_power, w_carrier, 0.22*w0, 0.08*w0, tt),  1      #FM_mask(mask_power, w_carrier, 0.22*w0, 0.08*w0, tt)
        #MODEL 2
        #F_mask, model    = AM_mask(mask_power, w_carrier, 0.5*w0, 0.1, tt),  2     
        #F_mask, model   = FM_mask(mask_power, w_carrier, 0.5*w0, 0.25*w0, tt),  2 

        if model == 1:
            sol1_m = solve_ivp(make_Hopf_dots_complex(F_female + F_mask), (0, num_cycles), z0, args=(mu, w0), t_eval=tt, method='RK45')
            Z1 = sol1_m.y[0]
            TE[i, trial] = Transfer_Entropy(Z1, F_female)
            
        if model == 2:
            sol_interm_m = solve_ivp(make_Hopf_dots_complex(F_female + F_male + F_mask), (0, num_cycles), z0, args=(mu, w0), t_eval=tt, method='RK45')
            sol2_m = solve_ivp(make_Hopf_dots_complex(sol_interm_m.y[0]), (0, num_cycles), z0, args=(mu, 0.5*w0), t_eval=tt, method='RK45')
            Z2 = sol2_m.y[0]
            TE[i, trial] = Transfer_Entropy(Z2, F_female)

if num_trials > 0:
    print(np.round( (time.time() - time1)/3600, 4), "hours")
    print("Transfer entropy")
    #print(np.array2string(np.round(TE, 5), separator=', ')[1:-1])
    print(TE.mean(axis=1))
    plt.figure()
    plt.plot(Carrier/(2*np.pi), TE.mean(axis=1), "-", color='black')
    plt.xlabel(r'$\Omega/\omega_0$')
    plt.ylabel("Transfer entropy (bits)")
    plt.ylim(0, 1.1*max(TE.mean(axis=1)))
    plt.xlim(0, 2)
    plt.figure()
    plt.plot(np.real(F_mask))
    plt.plot(np.imag(F_mask))
    plt.figure()
    plt.plot(np.real(F_mask), np.imag(F_mask))




#MODEL 1,  AM
TE1 = np.array([0.14260169, 0.14204211, 0.13938359, 0.13697541, 0.13410808,
       0.13242586, 0.12970477, 0.12729659, 0.12436982, 0.1215256 ,
       0.11987747, 0.1174229 , 0.10900303, 0.0923024 , 0.06355611,
       0.03647087, 0.05805032, 0.08895342, 0.10674572, 0.11475209,
       0.11741991, 0.12040231, 0.12341299, 0.12680229, 0.12897646,
       0.13078283, 0.13381008, 0.13695615, 0.13898323, 0.14039921,
       0.14282512])

#MODEL 2,  AM
TE2 = np.array([0.15914798, 0.15850827, 0.15793236, 0.15663792, 0.15630979,
       0.11651436, 0.08907762, 0.13744684, 0.1436305 , 0.14356867,
       0.14117724, 0.13591912, 0.1249651 , 0.10498668, 0.07954005,
       0.0714392 , 0.08588029, 0.12399638, 0.14659672, 0.15440886,
       0.15599596, 0.15949361, 0.15830901, 0.15569531, 0.15491941,
       0.15540591, 0.15846852, 0.15903397, 0.15997923, 0.15972709,
       0.15990453])

#MODEL 1,  FM
TE3 = np.array([0.1428807 , 0.14211165, 0.13922512, 0.13693922, 0.13405759,
       0.13230761, 0.13009041, 0.12781987, 0.12543283, 0.12263644,
       0.11926989, 0.11474852, 0.10828254, 0.09256962, 0.0654725 ,
       0.03566183, 0.06042574, 0.08991883, 0.10652686, 0.1118112 ,
       0.11687171, 0.12098013, 0.12466741, 0.12733581, 0.12919773,
       0.13074438, 0.13362057, 0.13691246, 0.13889371, 0.14041077,
       0.14256237])

#MODEL 2,  FM
TE4 = np.array([0.15894604, 0.15787261, 0.15704086, 0.15543371, 0.15352516,
       0.11826368, 0.0879755 , 0.13645512, 0.14366343, 0.14492266,
       0.14129693, 0.13728606, 0.12725289, 0.10778849, 0.08237227,
       0.06852762, 0.08899339, 0.12530378, 0.14717553, 0.15453001,
       0.1561732 , 0.15799656, 0.15691337, 0.15250725, 0.15109806,
       0.15149215, 0.15629269, 0.15866718, 0.15945222, 0.15985502,
       0.15986698])

Carrier_W = np.linspace(0.2, 1.8, 31)*2*np.pi / (2*np.pi)
clrs = ["#00AAFF", "#FF7300"]  #res 1Hopf,    res 2Hopf,  mask
legend_labels = ["M1, AM", "M2, AM", "M1, FM", "M2, FM"]
plt.figure(figsize=(4, 4), constrained_layout=True)
plt.plot(Carrier_W, TE1, "--", color=clrs[0], label=legend_labels[0])
plt.plot(Carrier_W, TE2, "--", color=clrs[1], label=legend_labels[1])
plt.plot(Carrier_W, TE3, "o-",  color=clrs[0], label=legend_labels[2], markersize=4)
plt.plot(Carrier_W, TE4, "o-",  color=clrs[1], label=legend_labels[3], markersize=4)
plt.xlabel(r'$\Omega/\omega_0$')
plt.ylabel("Transfer entropy (bits)")
plt.xlim(0, 2)
plt.ylim(0, 0.2)
plt.legend(loc='lower right')

plt.savefig("C:/Users/Justin/Desktop/Fig_test.pdf", dpi=300)

plt.show()
code.interact(local=locals())  #allows interaction with variables in terminal after







