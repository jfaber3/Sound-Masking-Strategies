import numpy as np
import matplotlib.pyplot as plt
import code


def Fig_Maker(fname1, fname2, xlabel1, xlabel2, scale_x1=1, scale_x2=1, save=False):
    fig = plt.figure(figsize=(5, 5))
    plt.subplots_adjust(left=0.15, right=0.62, bottom=0.11, top=0.97, hspace=0.35)
    ax1  = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
    ax2  = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)
    clrs = ["#00AAFF", "#FF7300"]  #res 1Hopf,    res 2Hopf,  mask
    line_styles = ["-", "--", ":"]
    fixed_param, sweep_param, TE1, TE2 = np.load(fname1, allow_pickle=True)
    print(fixed_param)
    for i in range(len(fixed_param)):
        ax1.plot(sweep_param/scale_x1, TE1[i, :], line_styles[i], color=clrs[0], label=legend_labels1[i])
        ax1.plot(sweep_param/scale_x1, TE2[i, :], line_styles[i], color=clrs[1], label=legend_labels2[i])
    fixed_param, sweep_param, TE1, TE2 = np.load(fname2, allow_pickle=True)
    print(fixed_param)
    for i in range(len(fixed_param)):
        ax2.plot(sweep_param/scale_x2, TE1[i, :], line_styles[i], color=clrs[0], label=legend_labels3[i])
        ax2.plot(sweep_param/scale_x2, TE2[i, :], line_styles[i], color=clrs[1], label=legend_labels4[i])
    ax1.set_xlabel(xlabel1)
    ax2.set_xlabel(xlabel2)
    ax1.set_ylabel("Transfer entropy (bits)")
    ax2.set_ylabel("Transfer entropy (bits)")
    ax1.set_ylim(0, y_limit)
    ax2.set_ylim(0, y_limit)
    #ax2.set_xticks([0.0, 0.1, 0.2, 0.3, np.sqrt(2)*0.3], ["0.0", "0.1", "0.2", "0.3", "$\sqrt{2P_t}$"])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save == True:
        plt.savefig("C:/Users/Justin/Desktop/Fig_test.pdf", dpi=300)


'''
#Noise Mask       (Hopf1 best 0.0365)   (Hopf2 best 0.0691)
y_limit = 0.18
fname1 = "C:/Users/Justin/Desktop/Auditory Masking/Manuscript Figs v1/TE_Data_noise_mask_p1_seed=12_trials=100.npy"
fname2 = "C:/Users/Justin/Desktop/Auditory Masking/Manuscript Figs v1/TE_Data_noise_mask_p2_seed=12_trials=60.npy"
xlabel1, xlabel2 = "$\omega/\omega_0$", "$\sigma_{\omega}/\omega_0$"
legend_labels1 = ["M1, $\sigma_{\omega} = 0$", "M1, $\sigma_{\omega} = 0.1\omega_0$", "M1, $\sigma_{\omega} = 0.2\omega_0$"]
legend_labels2 = ["M2, $\sigma_{\omega} = 0$", "M2, $\sigma_{\omega} = 0.1\omega_0$", "M2, $\sigma_{\omega} = 0.2\omega_0$"]
legend_labels3 = ["M1, $\omega = \omega_0$", "M1, $\omega = 1.1\omega_0$", "M1, $\omega = \omega_2$"]
legend_labels4 = ["M2, $\omega = \omega_0$", "M2, $\omega = 1.1\omega_0$", "M2, $\omega = \omega_2$"]
Fig_Maker(fname1, fname2, xlabel1, xlabel2, scale_x1=2*np.pi, scale_x2=2*np.pi, save=True)
plt.show()
'''

'''
#Two Tone Mask       (Hopf1 best 0.0360)     (Hopf2 best 0.0666)
y_limit = 0.1
fname1 = "C:/Users/Justin/Desktop/Auditory Masking/Manuscript Figs v1/TE_Data_TwoTone_mask_p1_seed=12_trials=50.npy"
fname2 = "C:/Users/Justin/Desktop/Auditory Masking/Manuscript Figs v1/TE_Data_TwoTone_mask_p2_seed=12_trials=100.npy"
xlabel1, xlabel2 = "$\Omega_2/\omega_0$", "$A_2/A_1$"
legend_labels1 = ["M1, $A_2/A_1 = 0.1$", "M1, $A_2/A_1 = 0.5$", "M1, $A_2/A_1 = 1$"]
legend_labels2 = ["M2, $A_2/A_1 = 0.1$", "M2, $A_2/A_1 = 0.5$", "M2, $A_2/A_1 = 1$"]
legend_labels3 = ["M1, $\Omega_2 = 0.975\omega_0$", "M1, $\Omega_2 = 1.25\omega_0$", "M1, $\Omega_2 = \omega_2$"]
legend_labels4 = ["M2, $\Omega_2 = 0.975\omega_0$", "M2, $\Omega_2 = 1.25\omega_0$", "M2, $\Omega_2 = \omega_2$"]
Fig_Maker(fname1, fname2, xlabel1, xlabel2, scale_x1=2*np.pi, scale_x2=1, save=True)
plt.show()
'''

'''
#Amplitude Modulation Mask      (Hopf1 best 0.0363)     (Hopf2 best 0.0676)
y_limit = 0.18
fname1 = "C:/Users/Justin/Desktop/Auditory Masking/Manuscript Figs v1/TE_Data_AM_mask_p1_seed=12_trials=60.npy"
fname2 = "C:/Users/Justin/Desktop/Auditory Masking/Manuscript Figs v1/TE_Data_AM_mask_p2_seed=12_trials=60.npy"
xlabel1, xlabel2 = "$\omega_{mod}/\omega_0$", "$A_{mod}$"
legend_labels1 = ["M1, $A_{mod} = 0.1$", "M1, $A_{mod} = 0.2$", "M1, $A_{mod} = \sqrt{2P_m}$"]
legend_labels2 = ["M2, $A_{mod} = 0.1$", "M2, $A_{mod} = 0.2$", "M2, $A_{mod} = \sqrt{2P_m}$"]
legend_labels3 = ["M1, $\omega_{mod} = 0.05\omega_0$", "M1, $\omega_{mod} = 0.3\omega_0$", "M1, $\omega_{mod} = \omega_2$"]
legend_labels4 = ["M2, $\omega_{mod} = 0.05\omega_0$", "M2, $\omega_{mod} = 0.3\omega_0$", "M2, $\omega_{mod} = \omega_2$"]
Fig_Maker(fname1, fname2, xlabel1, xlabel2, scale_x1=2*np.pi, scale_x2=1, save=True)
plt.show()
'''


#Frequency Modulation Mask      (Hopf1 best 0.0354)    (Hopf2 best 0.0672)
y_limit = 0.15
fname1 = "C:/Users/Justin/Desktop/Auditory Masking/Manuscript Figs v1/TE_Data_FM_mask_p1_seed=12_trials=50.npy"
fname2 = "C:/Users/Justin/Desktop/Auditory Masking/Manuscript Figs v1/TE_Data_FM_mask_p2_seed=12_trials=50.npy"
xlabel1, xlabel2 = "$\omega_{mod}/\omega_0$", "$A_{mod}/\omega_0$"
legend_labels1 = ["M1, $A_{mod} = 0.05\omega_0$", "M1, $A_{mod} = 0.1\omega_0$", "M1, $A_{mod} = 0.2\omega_0$"]
legend_labels2 = ["M2, $A_{mod} = 0.05\omega_0$", "M2, $A_{mod} = 0.1\omega_0$", "M2, $A_{mod} = 0.2\omega_0$"]
legend_labels3 = ["M1, $\omega_{mod} = 0.05\omega_0$", "M1, $\omega_{mod} = 0.2\omega_0$", "M1, $\omega_{mod} = 0.5\omega_0$"]
legend_labels4 = ["M2, $\omega_{mod} = 0.05\omega_0$", "M2, $\omega_{mod} = 0.2\omega_0$", "M2, $\omega_{mod} = 0.5\omega_0$"]
Fig_Maker(fname1, fname2, xlabel1, xlabel2, scale_x1=2*np.pi, scale_x2=2*np.pi, save=True)
plt.show()


'''
#LINEAR Frequency Modulation Mask      (Hopf1 best 0.0355)  (Hopf2 best 0.0681)
y_limit = 0.14
fname1 = "C:/Users/Justin/Desktop/Auditory Masking/Manuscript Figs v1/TE_Data_FM_linear_mask_p1_seed=12_trials=70.npy"
fname2 = "C:/Users/Justin/Desktop/Auditory Masking/Manuscript Figs v1/TE_Data_FM_linear_mask_p2_seed=12_trials=60.npy"
xlabel1, xlabel2 = "$f_{mod}$", "$A_{mod}/\omega_0$"
legend_labels1 = ["M1, $A_{mod} = 0.04\omega_0$", "M1, $A_{mod} = 0.07\omega_0$", "M1, $A_{mod} = 0.34\omega_0$"]
legend_labels2 = ["M2, $A_{mod} = 0.04\omega_0$", "M2, $A_{mod} = 0.07\omega_0$", "M2, $A_{mod} = 0.34\omega_0$"]
legend_labels3 = ["M1, $f_{mod} = 0.02$", "M1, $f_{mod} = 0.1$", "M1, $f_{mod} = 0.5$"]
legend_labels4 = ["M2, $f_{mod} = 0.02$", "M2, $f_{mod} = 0.1$", "M2, $f_{mod} = 0.5$"]
Fig_Maker(fname1, fname2, xlabel1, xlabel2, scale_x1=1, scale_x2=2*np.pi, save=True)
plt.show()
'''

'''
#POWER LAW Frequency Modulation Mask     (Hopf1 best 0.0355)  (Hopf2 best 0.0671)
y_limit = 0.12
fname1 = "C:/Users/Justin/Desktop/Auditory Masking/Manuscript Figs v1/TE_Data_FM_power_mask_p1_seed=12_trials=80.npy"
fname2 = "C:/Users/Justin/Desktop/Auditory Masking/Manuscript Figs v1/TE_Data_FM_power_mask_p2_seed=12_trials=80.npy"
xlabel1, xlabel2 = "$A_{mod}/\omega_0$", r'$\alpha$'
legend_labels1 = [r'M1, $\alpha = 0.5$', r'M1, $\alpha = 1$', r'M1, $\alpha = 2$']
legend_labels2 = [r'M2, $\alpha = 0.5$', r'M2, $\alpha = 1$', r'M2, $\alpha = 2$']
legend_labels3 = ["M1, $A_{mod} = 0.02\omega_0$", "M1, $A_{mod} = 0.04\omega_0$", "M1, $A_{mod} = 0.06\omega_0$"]
legend_labels4 = ["M2, $A_{mod} = 0.02\omega_0$", "M2, $A_{mod} = 0.04\omega_0$", "M2, $A_{mod} = 0.06\omega_0$"]
Fig_Maker(fname1, fname2, xlabel1, xlabel2, scale_x1=2*np.pi, scale_x2=1, save=True)
plt.show()
'''

'''
#SQUARE WAVE Frequency Modulation Mask      (Hopf1 best 0.0353)  (Hopf2 best 0.0737)
y_limit = 0.16
fname1 = "C:/Users/Justin/Desktop/Auditory Masking/Manuscript Figs v1/TE_Data_FM_square_mask_p1_seed=12_trials=50.npy"
fname2 = "C:/Users/Justin/Desktop/Auditory Masking/Manuscript Figs v1/TE_Data_FM_square_mask_p2_seed=12_trials=50.npy"
xlabel1, xlabel2 = "$f_{mod}$", "$A_{mod}/\omega_0$"
legend_labels1 = ["M1, $A_{mod} = 0.05\omega_0$", "M1, $A_{mod} = 0.1\omega_0$", "M1, $A_{mod} = 0.2\omega_0$"]
legend_labels2 = ["M2, $A_{mod} = 0.05\omega_0$", "M2, $A_{mod} = 0.1\omega_0$", "M2, $A_{mod} = 0.2\omega_0$"]
legend_labels3 = ["M1, $f_{mod} = 0.14$", "M1, $f_{mod} = 0.48$", "M1, $f_{mod} = 0.8$"]
legend_labels4 = ["M2, $f_{mod} = 0.14$", "M2, $f_{mod} = 0.48$", "M2, $f_{mod} = 0.8$"]
Fig_Maker(fname1, fname2, xlabel1, xlabel2, scale_x1=1, scale_x2=2*np.pi, save=True)
plt.show()
'''

code.interact(local=locals())  #allows interaction with variables in terminal after








