# the GWtoolkit module
import numpy as np
import scipy.integrate as integrate
######### constant ######
GMsun=1.32754125e26 # cgs unit
DKPC=3.086e+21 # cgs unit
c=3e10 #cgs
################################################# cosmology functions #####################################################
zhigh=100.

def F(Z):
# function for integral
    return integrate.quad(lambda z:1.0/np.sqrt(z**3+1),0,Z)[0]

def DM(z,OmegaL=0.683,h=0.7): 
# OmegaL is the fraction of dark energy density of the critical density. h is the reduced hubble constant.
    OmegaM=1.-OmegaL
    A=(OmegaL/OmegaM)**0.3333
    DH=3001.0/h # Mpc
    return A*DH/np.sqrt(OmegaL)*(F((z+1)/A)-F(1.0/A))
# Comoving distance as function of the redshift, result is in Mpc

def Dl(z,OmegaL=0.683,h=0.7): #luminous distance
    return (1+z)*DM(z,OmegaL,h)
# result is in Mpc

def EZ(z,OmegaL=0.683):
    OmegaM=1.-OmegaL
    return np.sqrt(OmegaM*(1+z)**3+OmegaL)

def Normal(zhigh=100.0,OmegaL=0.683,h=0.7):
# the normalization of the distribution. Normal*DH^3 is equivalent 
#to the total "comoving volumn" of the universe up to z=zhigh.
    DH=3001.0/h # Mpc
    result=integrate.quad(lambda z:DM(z,OmegaL,h)**2/(1+z)/EZ(z,OmegaL),0,zhigh)[0]/DH**2
    return result

def dVperdz(z,OmegaL=0.683,h=0.7):
# dV/dz, the comoving volumn element per red shift. 
#It is equivalent to the distribution of the redshift in the universe, given uniform comoving number density.
    OmegaM=1.-OmegaL
    A=(OmegaL/OmegaM)**0.3333
    return (A/np.sqrt(OmegaL)*(F((z+1)/A)-F(1.0/A)))**2/(EZ(z,OmegaL)*(1+z))
############################################### Monte Carlo sampling ######################################################
def drawzsample(N=1000):
# draw sample in the red-shift. The size of the sample is not exactly N, but some number close to it.
    X=np.logspace(-1,2,50)
    Y=[dVperdz(x) for x in X] # the un-normalized population
    upper=max(Y) # the peak value of the population distribution
    zwhole=np.random.uniform(0,zhigh,N)
    ywhole=np.random.uniform(0,upper,N)
    pwhole=np.interp(zwhole,X,Y)
    mask=(ywhole<pwhole)
    index=np.where(mask==True)[0]
    Nnew=round(N/(len(index))*N)
    zwhole=np.random.uniform(0,zhigh,Nnew)
    ywhole=np.random.uniform(0,upper,Nnew)
    pwhole=np.interp(zwhole,X,Y)
    mask=(ywhole<pwhole)
    index=np.where(mask==True)[0]
    len(index)
    zsampled=zwhole[index]
    return zsampled

def drawsample(N=1000,flag=0):
# draw sample in the multi-dimensional parameters space. 
#The size equals that of zsample, which is not exactly N, but someting close to this value.
    zsampled=drawzsample(N)
    Num=len(zsampled)
    Msampled=np.random.normal(loc=1.5,scale=0.3,size=Num) 
# Chirp mass before redshifted!! Gaussian distribution with mean mass 1.5 solar mass, varsqrt=0.5 solar mass
    cthsampled=np.random.uniform(1e-5,1,size=Num) # cos value of the polar angle in celestrial sphere.
    phisampled=np.random.uniform(0,6.283,size=Num)# azimuth angle in the celestrial sphere
    cosisampled=np.random.uniform(1e-5,1,size=Num)# cos value of the inclination angle
    psisampled=np.random.uniform(0,3.14159,size=Num)#polarization angle value.
    if flag==0: # give no subsample
        return [Msampled,cthsampled,phisampled,cosisampled,psisampled,zsampled]
    else :
        subM=[] # un-redshifted chirp mass
        subz=[] # subsample of redshift.
        for i in range(0,Num):
            D=Dl(zsampled[i])
            flag2=f(Msampled[i]*(1+zsampled[i]),D,cthsampled[i],phisampled[i],cosisampled[i],psisampled[i],rhocri=10)
            if flag2==1:
                subM.append(Msampled[i])
                subz.append(zsampled[i])
        return [Msampled,cthsampled,phisampled,cosisampled,psisampled,zsampled,subM,subz]
##########################################################################################################################
M0=1.31 #solar mass, red-shifted Chirp mass
D0=1 #Mpc
rho0=5.1e4 # the SNR, of the template_0 over the Einstein telescope noise.
#rhocri=1e5
def F1(cth,phi,psi):
    plus=-0.433*((1.0+cth**2)*np.sin(2.0*phi)*np.cos(2.0*psi)+2.0*cth*np.cos(2.0*phi)*np.sin(2.0*psi))
    cross=0.433*((1.0+cth**2)*np.sin(2.0*phi)*np.sin(2.0*psi)-2.0*cth*np.cos(2.0*phi)*np.cos(2.0*psi))
    return [plus,cross]

def rho(M,D,cth,phi,ci,psi):
    # M is the red-shifted chirp mass.
    Deff=D/np.sqrt((0.5*(1.0+ci**2))**2*F1(cth,phi,psi)[0]**2+ci**2*F1(cth,phi,psi)[1]**2)
    #result=M/M0*D0/D*np.sqrt((1+ci**2)**2*F1(cth,phi,psi)[0]**2+4.*ci**2*F1(cth,phi,psi)[1]**2)*rho0
    result=(M/M0)**0.8333*(D0/Deff)*rho0
    return result

def f(M,D,cth,phi,ci,psi,rhocri):
    # the flag function. If the calculated SNR is larger than the critirial, then the flag is 1, else 0.
    rhothis=rho(M,D,cth,phi,ci,psi)
    if rhothis>=rhocri:
        result=1.
    else:
        result=0.
    return result

def Percent(cri,N=1000):
    # return the value of <f(rho)>, together with its uncertainty.
    fraction=[]
    for i in range(0,10):
        Msampled,cthsampled,phisampled,cosisampled,psisampled,zsampled=drawsample(N)
        NM=len(Msampled)
        summ=0.
        for j in range(0,NM):
            D=Dl(zsampled[j])
            summ=summ+f(Msampled[j]*(1+zsampled[j]),D,cthsampled[j],phisampled[j],cosisampled[j],psisampled[j],cri)
            # here we include the chirp mass redshift effect.
            average=summ/(NM*1.0)
        fraction.append(average)
    return [np.mean(fraction),np.sqrt(np.var(fraction))]

####################################### import Einstein Telescope noise curve ############################################
Noise_ET=np.loadtxt('ET_D_data.txt')
freq=Noise_ET[:,0]
noise1=Noise_ET[:,1]
noise2=Noise_ET[:,2]
noise3=Noise_ET[:,3]
def Sn(f):
    return np.interp(f,freq,noise3**2)
#########################################################################################################################
######## A table#################
y101=0.6437
y102=0.1469
y103=-0.4098
y104=-0.1331
###############################################
y111=0.827
y112=-0.1228
y113=-0.03523
y114=-0.08172
###############################################
y121=-0.2706
y122=-0.02609
y123=0.1008
y124=0.1451
###############################################
y201=-0.05822
y202=-0.0249
y203=1.829
y204=-0.2714
###############################################
y211=-3.935
y212=0.1701
y213=-0.02017
y214=0.1279
###############################################
y301=-7.092
y302=2.325
y303=-2.87
y304=4.922

x102=-920.9
x112=492.1
x122=135
x202=6742
x212=-1053
x302=-1.34e4
###########################
x103=1.702e4
x113=-9566
x123=-2182
x203=-1.214e5
x213=2.075e4
x303=2.386e5
###########################
x104=-1.254e5
x114=7.507e4
x124=1.338e4
x204=8.735e5
x214=-1.657e5
x304=-1.694e6
###########################
x105=0
x115=0
x125=0
x205=0
x215=0
x305=0
###########################
x106=-8.898e5
x116=6.31e5
x126=5.068e4
x206=5.981e6
x216=-1.415e6
x306=-1.128e7
###########################
x107=8.696e5
x117=-6.71e5
x127=-3.008e4
x207=-5.838e6
x217=1.514e6
x307=1.089e7
###########################################################################################################################
def C(M1,M2,Deff,i,theta,phi,psi):
    M=M1+M2
#    factor=M*15.4787466e-6 #second;
    eta=M1*M2/M**2
#    result1=(GMsun*M)**0.83333/(2.0*D*DKPC*2.1433*c**1.5)*\
#    (0.208333*eta)**0.5*np.sqrt((1.0+np.cos(i)**2)**2*F1(theta,phi,psi)[0]**2+\
#                                4.0*np.cos(i)**2*F1(theta,phi,psi)[1]**2)   
    #Deff=D/np.sqrt((0.5*(1.0+np.cos(i)**2))**2*F1(theta,phi,psi)[0]**2+np.cos(i)**2*F1(theta,phi,psi)[1]**2)
    result=0.21*(GMsun*M*0.435)**0.83333/(Deff*DKPC)/c**1.5
    return result
Lorentz = lambda x1,x2,x3: 1.0/3.14159*0.5*x3/((x1-x2)**2+(0.5*x3)**2)
def gamma(f,M1,M2,chi1,chi2): # the input value is with unit Hz    
    M=M1+M2;
    eta=M1*M2/M**2;
    factor=M*15.4787466e-6 #second;
    ft=f*factor # convert the input frequency to a dimensionless one
    delta=(M1-M2)/M; # delta is independent with the total mass
    chi=(1+delta)*chi1*0.5+(1-delta)*chi2*0.5 #spin
    ###############################################
    mu01=1-4.455*(1-chi)**0.217+3.521*(1-chi)**0.26 # independent of the total mass
    mu02=(1-0.63*(1-chi)**0.3)*0.5
    mu03=(1-0.63*(1-chi)**0.3)*(1-chi)**0.45*0.25
    mu04=0.3236+0.04894*chi+0.01346*chi**2
    ###############################################
    f0=np.pi*1e-3
    f1=mu01+y101*eta+y111*eta*chi+y121*eta*chi**2+y201*eta**2+y211*eta**2*chi+y301*eta**3; 
    f2=mu02+y102*eta+y112*eta*chi+y122*eta*chi**2+y202*eta**2+y212*eta**2*chi+y302*eta**3;
    sigma=mu03+y103*eta+y113*eta*chi+y123*eta*chi**2+y203*eta**2+y213*eta**2*chi+y303*eta**3;
    f3=mu04+y104*eta+y114*eta*chi+y124*eta*chi**2+y204*eta**2+y214*eta**2*chi+y304*eta**3;  #these are indepened with the total mass
    ###############################################
    alpha2=-323.0/224+451.0*eta/168.0
    alpha3=(27.0/8-11.0*eta/6.0)*chi
    epsilon1=1.4547*chi-1.8897
    epsilon2=-1.8153*chi-1.6557
    ################################################    
    fp=ft/f1;
    v=ft**0.3333    
    Wm=(1+alpha2*v**2+alpha3*v**3)/(1+epsilon1*v+epsilon2*v**2)
    Wr=Wm*(f2/f1)**(-2.0/3)*(1+epsilon1*v+epsilon2*v**2)/Lorentz(f2,f2,sigma)
    if (ft<f1) and (ft>f0):
        result=(ft/factor)**(-7.0/6)*(1+alpha2*v**2+alpha3*v**3);
    elif (ft>=f1) and (ft<f2):
        result=Wm*(f1/factor)**(-7.0/6)*fp**(-2.0/3)*(1+epsilon1*v+epsilon2*v**2);
    elif ft>=f2 and ft<f3:
        result=Wr*(f1/factor)**(-7.0/6)*Lorentz(ft,f2,sigma)
    else: 
        result=0
    return [result,f1/factor,f2/factor,f3/factor]
###########################################################################################################################
def rho0M_integrate(Mchirp,D0=1e3): #input red-shifted chirp mass (solar mass), give you the standard SNR
    m0=Mchirp/0.87; # from chirp mass to individual mass of an equal mass binary.
    factor=2.0*m0*15.4787466e-6 #second;
    f0=np.pi*1e-3
    delta=0; # delta is independent with the total mass
    chi=0;
    eta=0.25
    def INTT(y): 
        fp=np.exp(y)
        if fp/factor>min(freq)+2.:
            result=fp*gamma(fp/factor,m0,m0,chi,chi)[0]**2/Sn(fp/factor)
        else:
            result=0
        return result*1e-48
    ###############################################
    mu01=1-4.455*(1-chi)**0.217+3.521*(1-chi)**0.26 # independent of the total mass
    mu02=(1-0.63*(1-chi)**0.3)*0.5
    mu03=(1-0.63*(1-chi)**0.3)*(1-chi)**0.45*0.25
    mu04=0.3236+0.04894*chi+0.01346*chi**2
    ###############################################
    f3=mu04+y104*eta+y114*eta*chi+y124*eta*chi**2+y204*eta**2+y214*eta**2*chi+y304*eta**3;  
    ylow=max(np.log(f0),np.log(min(freq*factor)))
    yhigh=np.log(f3)    
    rhosq=integrate.quad(INTT,ylow,yhigh)
    rho=np.sqrt(4.0*C(m0,m0,D0,0,0,0,0)**2*rhosq[0]*1e48/factor)
    #rho=np.sqrt(rhosq)
    #print(ylow,yhigh)
    #rho=np.sqrt(4.0*C(m0,m0,D0,0,0,0,0)**2*rhosq[0]*1e48)
    return rho
###########################################################################################################################
Mz=np.linspace(3.,800.,50);
RHO=[rho0M_integrate(mz) for mz in Mz];
def rho0M(Mchirp):
	return np.interp(Mchirp,Mz,RHO)
###############################################################################################################
#def subsample(cri=10):
#    # give you the list of detected sources.
#    subM=[] # un-redshifted chirp mass
#    subz=[] # subsample of redshift.
#
#    for i in range(0,NM):
#        D=Dl(zsampled[i])
#        flag=f(Msampled[i]*(1+zsample[i]),D,cthsampled[i],phisampled[i],cosisampled[i],psisampled[i],cri)
#        if flag==1:
#            subM.append(Msampled[i])
#            subz.append(zsample[i])
#    return [subM,subz]


