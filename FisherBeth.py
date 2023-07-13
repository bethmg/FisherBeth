# defines all the functions for beth's fisher code, beth 2023
#############################################################
import numpy as np

cosmo_labels=['fiducial', 'h_m', 'h_p', 'ns_m', 'ns_p', 'Ob2_m', 'Ob2_p', 'Om_m', 'Om_p', 's8_m', 's8_p']

""" creates a dictionary for the peak of the PDFs using a CDF cut between 0.03 and 0.9. 'cov' extra cosmo holds the 1000 reals PDFs for the covariance  """
zlist=[0.0,0.5,1.0]
Rlistm=[25,30]
def PeakDict(PDF_dict,PDF_forcov_dict,CosM=False):
    print('Building PDF dictionary')
    if CosM==True:
		cosmo_labels.append(['b1_m','b1_p'])
    peak_z={}
    for z in zlist:
        print('z =',z,end=': ')
        peak_cos={}
        chosen_indices={}
        for cosmo in cosmo_labels:
            peak_R={}
            if  cosmo=='fiducial':
                peak_forcov_R={}
                bins_R={}
            for R in Rlistm:
                PDF=PDF_dict[z][cosmo][R]
                if cosmo=='fiducial':
                    PDF_forcov=PDF_forcov_dict[z][cosmo][R]
                    CDF=np.cumsum(np.mean(PDF_forcov,axis=1))
                    # chosen values for Fisher paper
                    p_value_min = 0.03
                    p_value_max = 0.9
                    chosen_indices[R] = np.where((CDF > p_value_min)&(CDF < p_value_max))[0]      
                    bins_R[R]=chosen_indices[R]
                    peak_forcov_R[R]=PDF_forcov[chosen_indices[R]]
                if CosM==True:
                    peak_R[R]=PDF[chosen_indices[R]]
                else:
                    peak_R[R]=np.mean(PDF[chosen_indices[R]],axis=1)[:100]
            peak_cos[cosmo]=peak_R
            peak_cos['bins']=bins_R
            print(cosmo,end='...')            
        peak_cos['cov']=peak_forcov_R
        print('cov',end='...')
        
        print('Done!')
        peak_z[z]=peak_cos
    return peak_z

""" returns covariance or correlation matrix for PDFs and/or power spectra """    
# must define these before using functions:
PDF_dict={} # define using peak dict
Pk_dict={} # in July23 these are already in the PeakDict format with kmax=0.2

# if data vector including PDF call func with list of scales e.g. Rlist=[25,30]
# if data vector incl Pk, Pk=True

def Cov_Matrix(z,Rlist=None,Pk=False):
    N_sims=1000
    data_vector=np.empty([0,N_sims])
    if Rlist!=None:
        for R in Rlist:
            # data_vector=np.concatenate((data_vector,PDF_forcov_dict[z][R]))
            data_vector=np.concatenate((data_vector,PDF_dict[z]['cov'][R]))
    if Pk==True:
        data_vector=np.concatenate((data_vector,Pk_dict[z]['cov']))
    cov_matrix=np.cov(data_vector)
    return(cov_matrix)

def Corr_Matrix(z,Rlist=None,Pk=False):
    N_sims=1000
    data_vector=np.empty([0,N_sims])
    if Rlist!=None:
        for R in Rlist:
            # data_vector=np.concatenate((data_vector,PDF_forcov_dict[z][R]))
            data_vector=np.concatenate((data_vector,PDF_dict[z]['cov'][R]))
    if Pk==True:
        data_vector=np.concatenate((data_vector,Pk_dict[z]['cov']))
    corr_matrix=np.corrcoef(data_vector)
    return(corr_matrix)


""" differentiates PDF of Pk wrt parameter """
# note: properly implement b1 or this'll break!!
def Deriv(param,z,R=None,Pk=False):
    param_labels_lcdm=['Om','s8','Ob2','h','ns','b1']
    dlt_theta={'s8':0.015, 'Om':0.01, 'Ob2':0.002, 'ns':0.02, 'h':0.02, 'Mnu':[0.1,0.2,0.4],'b1':0.05}
    if not param in param_labels_lcdm:
        return 'param not supported'
    if param=='b1': # not sure why need this bit? 
        vec_p=PDF_dict[z][param+'_p'][R][:100]
        vec_m=PDF_dict[z][param+'_m'][R][:100]
        vec_deriv=(vec_p-vec_m)/(2*dlt_theta[param])
    else:
        if R!=None:
            vec_p=PDF_dict[z][param+'_p'][R]
            vec_m=PDF_dict[z][param+'_m'][R]
        elif Pk==True:
            vec_p=Pk_dict[z][param+'_p']
            vec_m=Pk_dict[z][param+'_m']   
        vec_deriv=(vec_p-vec_m)/(2*dlt_theta[param])
    return vec_deriv
    

""" differentiates data vector inc. PDF and/or Pk wrt parameter """
PDF_dict={}
Pk_dict={}
def Deriv_Vector(z,param,Rlist=None,Pk=False):
    vec=np.array([])
    # for param in param_labels:
    if Rlist!=None:
        for Rval in Rlist:
            vec=np.concatenate((vec,Deriv(param,z,R=Rval)))  # dict must be already specified
    if Pk==True:
        vec=np.concatenate((vec,Deriv(param,z,Pk=True)))
    return vec
          
    
""" creates fisher matrix and parameter covariance matrix"""
PDF_dict={}
Pk_dict={} 
def Fisher(zlist,param_labels,Rlist=None,Pk=False):
    N_sims=1000
    fisher=np.zeros([len(param_labels),len(param_labels)])
    for z in zlist:
        cov_matrix=Cov_Matrix(z,Rlist,Pk)
        hartlap=float(N_sims - 2 - len(cov_matrix[:,0]))/float(N_sims - 1)
        derivs_vector={}
        for param in param_labels:
            derivs_vector[param]=Deriv_Vector(z,param,Rlist,Pk)
        inverse_cov=np.linalg.inv(cov_matrix)*hartlap
        fisher_z=np.zeros([len(param_labels),len(param_labels)])
        for alpha in range(0,len(cov_matrix[:,0])):
            for beta in range(0,len(cov_matrix[:,0])):
                for i,param_i in enumerate(param_labels):
                    for j,param_j in enumerate(param_labels):
                        fisher_z[i,j] += derivs_vector[param_i][alpha]*inverse_cov[alpha,beta]*derivs_vector[param_j][beta]
        fisher+=fisher_z                
    return(fisher)

def Param_Covariance(zlist,param_labels,Rlist=None,Pk=False):
    mat=np.linalg.inv(Fisher(zlist,param_labels,Rlist,Pk))
    return mat


""" plots derivs in a bunch of ways """
PDF_dict={}
def DerivPlot(PDF_dict,title,save=(False,None),savetitle=None):
    
    param_labels_lcdm=['Om','s8','Ob2','h','ns']
    labels={'s8':'$\sigma_8$', 'Om':'$\Omega_m$', 'Ob2':'$\Omega_b$', 'ns':'$n_s$', 'h':'$h$', 'Mnu':r'$M_\nu$'}
    zlist=[0.0,0.5,1.0]
    linestylesm=['-','--',':']
    color_R=cm.tab10(np.linspace(1,0,10))[::-1]
    Rlistm=[25,30]
    for Rval in Rlistm:
        plt.figure()
        for z_i,z in enumerate(zlist):
            bins=PDF_dict[z]['bins'][Rval]
            plt.axhline(y=0,color='black',linestyle='-',linewidth=0.8)
            for param_i,param in enumerate(param_labels_lcdm):
                labels_z={0.0:r'$\theta$='+labels[param],0.5:'_nolegend_',1.0:'_nolegend_'}
                plt.plot(bins,Deriv(param,z,R=Rval),label=labels_z[z],linestyle=linestylesm[z_i],color=color_R[param_i])
            # plt.plot(range(len(halo_PDF_b1deriv[z][R])),halo_PDF_b1deriv[z][R],label=r'CosMomentum, $\theta=b_1$')
            plt.plot(bins,PDF_dict[z]['fiducial'][R],color='grey',linestyle=linestylesm[z_i],label='$z=$'+str(z)+', fiducial PDF')

        plt.legend(fontsize=10,ncol=2)
        plt.ylabel(r'$\partial P(N_h)/\partial \theta$')
        plt.xlabel(r'$N_h$')
        plt.xlim(bins[0],bins[-1])
        plt.ylim(-0.15,0.20)
        plt.title(title+r', $R='+str(Rval)+'\mathrm{Mpc}/h$, CDF$\in$['+str(p_value_min)+','+str(p_value_max)+']')   
        if save==(True,Rval):
            if savetitle!=None:
                SaveBethFig('paramderivs_'+savetitle+'_LCDM_R='+str(Rval),talk=True)
            else:
                print('Pls add savetitle')
                
def DerivPlot_z(PDF_dict,title,save=(False,None),savetitle=None):
    param_labels_lcdm=['Om','s8','Ob2','h','ns']
    labels={'s8':'$\sigma_8$', 'Om':'$\Omega_m$', 'Ob2':'$\Omega_b$', 'ns':'$n_s$', 'h':'$h$', 'Mnu':r'$M_\nu$'}
    zlist=[0.0,0.5,1.0]
    linestylesm=['-','--',':']
    color_R=cm.tab10(np.linspace(1,0,10))[::-1]
    Rlistm=[25,30]
    Rshow=30
    for z_i,z in enumerate(zlist):
        for R in Rlistm:
            bins=PDF_dict[z]['bins'][R]
            plt.figure()
            plt.axhline(y=0,color='black',linestyle='-',linewidth=0.8)
            for param_i,param in enumerate(param_labels_lcdm):
                plt.plot(bins,Deriv(param,z,R=R),label=r'$\theta$='+labels[param],color=color_R[param_i])
            # plt.plot(range(len(halo_PDF_b1deriv[z][R])),halo_PDF_b1deriv[z][R],label=r'CosMomentum, $\theta=b_1$')
            plt.plot(bins,PDF_dict[z]['fiducial'][R],color='grey',linestyle=':',label='$z=$'+str(z)+', fiducial PDF')

            plt.legend(fontsize=10,ncol=2)
            plt.ylabel(r'$\partial P(N_h)/\partial \theta$')
            plt.xlabel(r'$N_h$')
            plt.xlim(bins[0],bins[-1])
            # plt.ylim(-0.15,0.20)
            plt.title(title+r', $z=$'+str(z)+', $R='+str(R)+'\mathrm{Mpc}/h$, CDF$\in$['+str(p_value_min)+','+str(p_value_max)+']')   
            if save==(True,R):
                if savetitle!=None:
                    SaveBethFig('paramderivs_'+savetitle+'_LCDM_R='+str(R),paper=True)
                else:
                    print('Pls add savetitle')                

Pk_dict={}
def DerivPlotPk(Pk_dict,zlist):
    param_labels_lcdm=['Om','s8','Ob2','h','ns']
    labels={'s8':'$\sigma_8$', 'Om':'$\Omega_m$', 'Ob2':'$\Omega_b$', 'ns':'$n_s$', 'h':'$h$', 'Mnu':r'$M_\nu$'}
    linestylesm=['-','--',':']
    color_R=cm.tab10(np.linspace(1,0,10))[::-1]
    plt.figure()
    plt.axhline(y=0,color='black',linestyle='-',linewidth=0.8)
    for z_i,z in enumerate(zlist):
        bins=Pk_dict[z]['bins']
        for param_i,param in enumerate(param_labels_lcdm):
            deriv=Deriv(param,z,Pk=True)
            if z_i==0:
                plt.plot(bins,deriv,label=r'$\theta=$'+labels[param],color=color_R[param_i],linestyle=linestylesm[z_i])
            else:
                plt.plot(bins,deriv,color=color_R[param_i],linestyle=linestylesm[z_i])
        plt.plot(bins,Pk_dict[z]['fiducial'],color='black',linestyle=linestylesm[z_i],label='$z=$'+str(z)+', fiducial $P(k)$')
        # plt.plot([],[],label='$z=$'+str(z),color='black')

    plt.legend(ncol=2)
    plt.xlabel('$k$')
    plt.ylabel(r'$\partial P(k)/\partial \theta$')
    plt.title(r'$P(k)$ param derivatives')#, $k\in[0.05,0.2]h$/Mpc')
