# the GWtoolkit module
import numpy as np
import scipy.integrate as integrate
zhigh=100.

def F(Z):
    return integrate.quad(lambda z:1.0/np.sqrt(z**3+1),0,Z)[0]

def DM(z,OmegaL=0.683,h=0.7): # OmegaL is the fraction of dark energy density of the critical density. h is the reduced hubble constant.
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
# the normalization of the distribution. Normal*DH^3 is equivalent to the total "comoving volumn" of the universe up to z=zhigh.
    DH=3001.0/h # Mpc
    result=integrate.quad(lambda z:DM(z,OmegaL,h)**2/(1+z)/EZ(z,OmegaL),0,zhigh)[0]/DH**2
    return result

def dVperdz(z,OmegaL=0.683,h=0.7):
# dV/dz, the comoving volumn element per red shift. It is equivalent to the distribution of the redshift in the universe, given uniform comoving number density.
    OmegaM=1.-OmegaL
    A=(OmegaL/OmegaM)**0.3333
    return (A/np.sqrt(OmegaL)*(F((z+1)/A)-F(1.0/A)))**2/(EZ(z,OmegaL)*(1+z))

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
# draw sample in the multi-dimensional parameters space. The size equals that of zsample, which is not exactly N, but someting close to this value.
    zsampled=drawzsample(N)
    Num=len(zsampled)
    Msampled=np.random.normal(loc=1.5,scale=0.3,size=Num) # Chirp mass before redshifted!! Gaussian distribution with mean mass 1.5 solar mass, varsqrt=0.5 solar mass
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


