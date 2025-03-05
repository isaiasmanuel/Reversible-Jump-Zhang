#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:37:30 2024

@author: isaias
"""
########Paqueterias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import mpmath as mp
import scipy.special as sc


def Mmvn(a,mu,lam,w):
    M=len(mu)
    Sigmam=np.zeros((M,2,2))
    for i in range(M):
        Sigmam[i,:,:]=A@np.diag(1/lam[i])@A.T

    detSigma=np.zeros(M)

    for i in range(M):
        detSigma[i]=np.linalg.det(Sigmam[i,:,:])
    densidad=0
    for i in range(M):
        densidad+=w[i]*np.exp(-(a-mu[i])@Sigmam[i,:,:]@np.reshape(a-mu[i], (2,1))/2)/(2*np.pi*detSigma[i])

    return densidad[0]




######Generamos los datos
def SimulaMezcla(grupo):
    if grupo==0:
        sim=sp.stats.norm.rvs(size=(1,2),loc=(5,5))

    elif grupo==1:
        # sim=sp.stats.norm.rvs(size=(1,2),loc=(2,2))
        sim=np.reshape(sp.stats.multivariate_normal.rvs(size=(1,1),mean=(5,-5), cov= np.array(((5,3.7),(3.7,5))) ),(1,2))
        
    elif grupo==2:
          # sim=sp.stats.norm.rvs(size=(1,2),loc=(2,2))
           sim=np.reshape(sp.stats.multivariate_normal.rvs(size=(1,1),mean=(10,-5), cov= np.array(((5,-3.7),(-3.7,5))) ),(1,2))
         
        
    else :
        sim=sp.stats.norm.rvs(size=(1,2),loc=(2,-15))

    return sim


n=1000
# Grupo=np.random.choice((0,1,2),size=n,p=(0.2,0.6,0.2))

Grupo=np.random.choice((0,1,2,3),size=n,p=(0.25,0.25,0.25,0.25))

X=np.apply_along_axis(SimulaMezcla, 0, np.reshape(Grupo, (1,len(Grupo))))
X=np.transpose(X[0])

plt.scatter(X[:,0],X[:,1])
plt.show()


############# Hiperparametros de las priors

A=np.array([[1,0],[0,1]]) ####Eigenvalores
rm=1                      ###Hiperparametro lambda
nu=np.array((4,4))        ###Hiperparametro xi
escala=10
rho=10                    ###Hiperparametro tau
greekzeta=1               ###Hiperparametro l
delta=1

############# Parametros iniciales
###Puntos iniciales del alogritmo
M=3 #Inicio Mezcla

w=np.ones(M)/M
mu=np.ones((M,2))+np.random.normal(size=(M,2))*5
mu=mu[mu[:,0].argsort(),:]
lam=np.ones((M,2))  #Estoy almacenando laminv
xi=np.ones((M,2))
tau=np.ones(M)
linv=np.ones((M,2))
Z=np.random.choice(np.arange(M), size=n)

#######Priors hyperparametros
priorXi=sp.stats.multivariate_normal(mean=nu,cov=rho**2*np.diag((1,1)))
priorTau=sp.stats.gamma(a=0.5,scale=1/(rho**-2))
priorlinv=sp.stats.gamma(a=0.5,scale=1/(greekzeta/2))


################################################################################################################################################################################
whist=[]
muhist=[]
lamhist=[]
xihist=[]
tauhist=[]
linvhist=[]
Zhist=[]
Mhist=[]

whist.append(w)
muhist.append(mu)
lamhist.append(lam)
xihist.append(xi)
tauhist.append(tau)
linvhist.append(linv)
Zhist.append(Z)
Mhist.append(M)

for q in np.arange(10000):

    w=np.copy(whist[-1])
    mu=np.copy(muhist[-1])
    lam=np.copy(lamhist[-1])
    xi=np.copy(xihist[-1])
    tau=np.copy(tauhist[-1])
    linv=np.copy(linvhist[-1])
    Z=np.copy(Zhist[-1])

    PasoProb=(1,1,1,1,1)
    # PasoProb=(0,0,0,0,1)
    # PasoProb=(1,1,1,1,0)
    step=np.random.choice(np.arange(5), p=PasoProb/np.sum(PasoProb))


    XZ=(pd.DataFrame((X[:,0],X[:,1],np.copy(Z)))).transpose()
    XZ.columns=("x","y","g")


    ############Actualizar
    M=len(mu)
    Sigmam=np.zeros((M,2,2))

    for i in range(M):
        Sigmam[i,:,:]=A@np.diag(1/lam[i])@A.T

    detSigma=np.zeros(M)

    for i in range(M):
        detSigma[i]=np.linalg.det(Sigmam[i,:,:])


    ni=np.zeros(M)
    for i in range(M):
        ni[i]=np.sum(XZ["g"]==i)

    # np.sum(ni)
    # ni=np.unique(Z, return_counts=True)[1]

    ############stepa
    if step==0:

        w=np.random.dirichlet(delta+ni)

    ############stepb
    if step==1 and M<20:
        # Xbar=XZ.groupby('g').sum()
        Xbar=np.zeros((M,2))
        for i in range(M):
            Xbar[int(i),0]=np.sum(XZ[XZ["g"]==i]["x"])
            Xbar[int(i),1]=np.sum(XZ[XZ["g"]==i]["y"])

        # Xbar
        # Xbar=Xbar.to_numpy()

        # Xbar=Xbar#/np.reshape(ni,(5,1))


        Sm=np.zeros((M,2,2))

        for i in range(M):
            Sm[i,:,:]=(X[Z==i]-mu[i]).T@(X[Z==i]-mu[i])

        barxi=(np.reshape(tau,(M,1))*xi+Xbar)/np.reshape((tau+ni),(M,1))


        for i in range(M):
            mu[i,:]=sp.stats.multivariate_normal.rvs(size=1, mean=barxi[i],cov=Sigmam[i]/(ni[i]+tau[i]))

        for i in range(M):
            for j in range(2):
                lam[i,j]=1/np.random.gamma(shape=(rm+ni[i]+1)/2, scale=1/((linv[i,j]+A[:,j]@(tau[i]*np.reshape((mu[i]-xi[i]),(2,1))@np.reshape((mu[i]-xi[i]),(1,2))+Sm[i])@(A[:,j]).T)/2) )


        reorder=mu[:,0].argsort()
        mu=mu[reorder,:]
        lam=lam[reorder,:]
        w=w[reorder]
        xi=xi[reorder,:]
        tau=tau[reorder]
        linv=linv[reorder,:]

        cont=0
        for p in reorder:
            Z[Z==p]=cont+M
            cont+=1

        Z=Z-M
        # print(np.unique(Z))

    ############stepc
    if step==2:
        Sigmainvm=np.copy(Sigmam)
        for i in range(M):
            Sigmainvm[i,:,:]=np.linalg.inv(Sigmam[i,:,:])

        for i in range(len(X)):
            pns=np.zeros(M)
            for j in range(M):
                pns[j]= (w[j]/np.sqrt(detSigma[j])*np.exp(-1/2* (X[i,:]-mu[j]) @Sigmainvm[j,:,:]@np.reshape((X[i,:]-mu[j]),(2,1))))[0]
            pns=pns/np.sum(pns)
            
            Z[i]=np.sum(np.cumsum(pns)<np.random.uniform(size=1))

    # ############stepd
    if step==3:
        i=0
        Sigmainvm=np.copy(Sigmam)
        for i in range(M):
            Sigmainvm[i,:,:]=np.linalg.inv(Sigmam[i,:,:])

        hatsigmam=np.copy(Sigmam)
        for i in range(M):
            hatsigmam[i,:,:]=rho**(-2)*np.diag(np.ones(2))+tau[i]*Sigmainvm[i,:,:]


        for i in range(M):
            maux=np.linalg.inv(hatsigmam[i])@(rho**(-2)*nu+tau[i]*Sigmainvm[i,:,:]@mu[i])
            xi[i]=sp.stats.multivariate_normal.rvs(size=1, mean=maux,cov=np.linalg.inv(hatsigmam[i]))

        for i in range(M):
            tau[i]=np.random.gamma(shape=(X.shape[1]+1)/2, scale=1/((rho**(-2)+(mu[i,:]-xi[i,:])@Sigmainvm[i]@np.reshape((mu[i,:]-xi[i,:]),(2,1)))/2) )[0]

        for i in range(M):
            for j in range(2):
                linv[i,j]=np.random.gamma(shape=(1+rm)/2,scale=1/((lam[i,j]+greekzeta)/2))
        # print(step)


    if step<4:
        whist.append(w)
        muhist.append(mu)
        lamhist.append(lam)
        xihist.append(xi)
        tauhist.append(tau)
        linvhist.append(linv)
        Zhist.append(Z)
    

###############



###############







    ############stepe
    if step==4:
        ps=0.5
        split=np.random.choice((0,1),p=(ps,1-ps))
        #####split
        if split==1:
            

# UTIL PARA PRUEBAS

# ############################################################################################################################################
#             w=whist[-1]
#             mu=muhist[-1]
#             lam=lamhist[-1]
#             xi=xihist[-1]
#             tau=tauhist[-1]
#             linv=linvhist[-1]
#             Z=Zhist[-1]

#             step=np.random.choice(np.arange(5))


#             XZ=(pd.DataFrame((X[:,0],X[:,1],np.copy(Z)))).transpose()
#             XZ.columns=("x","y","g")


#             ############Actualizar
#             M=len(mu)
#             Sigmam=np.zeros((M,2,2))

#             for i in range(M):
#                 Sigmam[i,:,:]=A@np.diag(1/lam[i])@A.T

#             detSigma=np.zeros(M)

#             for i in range(M):
#                 detSigma[i]=np.linalg.det(Sigmam[i,:,:])


#             ni=np.zeros(M)
#             for i in range(M):
#                 ni[i]=np.sum(XZ["g"]==i)
#             # print(ni)
# ############################################################################################################################################



            #split
            Sigmainvm=np.copy(Sigmam)
            for i in range(M):
                Sigmainvm[i,:,:]=np.linalg.inv(Sigmam[i,:,:])


            alpha=np.random.beta(1, 1)
            betad=np.random.beta(1, 1,size=2)
            ud=np.random.choice((-1,1),size=2)*np.random.beta(2, 2,size=2)
            auxXi=np.random.normal(0, 1,size=2)
            # auxTau=np.random.beta(1, 1)
            # auxlinv=np.random.uniform(0,0.01,size=2)
            
            auxTau=sp.stats.expon.rvs( scale=escala)
            auxlinv=sp.stats.expon.rvs( scale=escala,size=2)
            


            update=np.random.choice(range(M))

            waux1=w[update]*alpha
            waux2=w[update]*(1-alpha)


            mumovement=np.sqrt(waux2/waux1)*(np.sqrt(1/lam[update][0])*ud[0]*A[:,0]+np.sqrt(1/lam[update][1])*ud[1]*A[:,1])

            muaux1=mu[update]-mumovement
            muaux2=mu[update]+mumovement


            lamaux1=1/(betad*(1-ud**2)*w[update]/waux1*(1/lam[update,:]))
            lamaux2=1/((1-betad)*(1-ud**2)*w[update]/waux2*(1/lam[update,:]))

            mu=np.vstack((mu[:update],muaux1,muaux2,mu[update+1:]))

            M=len(mu)

            lam=np.vstack((lam[:update],lamaux1,lamaux2,lam[update+1:]))

            w=np.hstack((w[:update],waux1,waux2,w[update+1:]))
            ###############Inventado

            ###
            # Z[Z>update]=Z[Z>update]+1
            # Z[Z==update]=np.random.choice((update,update+1),size=np.sum(Z==update))

            M=len(mu)
            SigmamNueva=np.zeros((M,2,2))

            for i in range(M):
                SigmamNueva[i,:,:]=A@np.diag(1/lam[i])@A.T

            detSigmaNueva=np.zeros(M)

            for i in range(M):
                detSigmaNueva[i]=np.linalg.det(SigmamNueva[i,:,:])


            SigmainvmNueva=np.copy(SigmamNueva)
            for i in range(M):
                SigmainvmNueva[i,:,:]=np.linalg.inv(SigmamNueva[i,:,:])
            
            Z[Z>update]=Z[Z>update]+1
            Zaux=np.zeros(np.sum([Z==update]))-1
            for i in range(np.sum(Z==update)):
                pns=np.zeros(M)
                for j in [update,update+1]:
                    pns[j]=(w[j]/np.sqrt(detSigmaNueva[j])*np.exp(-1/2* (X[Z==update][i,:]-mu[j]) @SigmainvmNueva[j,:,:]@np.reshape((X[Z==update][i,:]-mu[j]),(2,1))))[0]
                pns=pns/np.sum(pns)
                # print(pns)
                Zaux[i]=np.sum(np.cumsum(pns)<np.random.uniform(size=1))
            Z[Z==update]=np.copy(Zaux)
            

            niNuevo=np.zeros(M)
            for i in range(M):
                niNuevo[i]=np.sum(Z==i)
            
            len(Zaux)

            ###


            xi=np.vstack((xi[:update+1,:],xi[update,:]  ,xi[update+1:,:]))
            xi[update]=xi[update,:]+auxXi
            xi[update+1]=xi[update+1]-auxXi

            tau=np.hstack((tau[:update+1], tau[update],tau[update+1:]))
            # tau[update]=tau[update]+auxTau
            # tau[update+1]=tau[update+1]-auxTau
            tau[update]=tau[update]
            tau[update+1]=tau[update+1]/auxTau


            linv=np.transpose(np.vstack(
            (np.hstack((linv[:update+1,0],linv[update,0],linv[update+1:,0])),
            np.hstack((linv[:update+1,1],linv[update,1],linv[update+1:,1])))))
            
            linv[update]=linv[update]
            linv[update+1]=linv[update+1]/auxlinv
            
            xi[update]
            xihist[-1][update]
            
            x = np.linspace(np.min(X[:,0]),np.max(X[:,0]),50)
            y = np.linspace(np.min(X[:,1]),np.max(X[:,1]),50)
            Xg,Yg = np.meshgrid(x,y)
    
            # pdf = np.zeros(Xg.shape)
            # for i in range(Xg.shape[0]):
            #     for j in range(Xg.shape[1]):
            #         Obs=np.array((Xg[i,j], Yg[i,j]))
            #         pdf[i,j] = Mmvn(Obs,mu,lam,w)
    
            # plt.plot()
            # plt.contourf(Xg, Yg, pdf, cmap='viridis',levels=1000)
            # plt.scatter(X[:,0],X[:,1],alpha=0.1)
            # plt.scatter(mu[:,0],mu[:,1],c="red",s=50,alpha=0.5)
            # plt.title(len(mu))
            # plt.colorbar()
            # plt.show()



            try :
                r0=priorXi.pdf(x=xi[update])*priorXi.pdf(x=xi[update+1])/priorXi.pdf(x=xihist[-1][update])
                r0*=mp.fprod((priorTau.pdf(tau[update]),priorTau.pdf(tau[update+1])))/priorTau.pdf(x=tauhist[-1][update])
                r0*=np.prod(priorlinv.pdf(linv[update]))*np.prod(priorlinv.pdf(linv[update+1]))/np.prod(priorlinv.pdf(linvhist[-1][update]))
                
                Zaux
                [Zhist[-1]==update]
                Indicadora0=np.zeros(M-1)
                Indicadora1=np.zeros(M)
                Indicadora2=np.zeros(M)
                Indicadora0[update] = 1
                Indicadora1[update] = 1
                Indicadora2[update+1] =1
                
                            
                r2=mp.fprod(np.apply_along_axis(lambda x: Mmvn(x,mu,lam,Indicadora1),1,X[Z==update]))*mp.fprod(np.apply_along_axis(lambda x: Mmvn(x,mu,lam,Indicadora2),1,X[Z==update+1]))/mp.fprod(np.apply_along_axis(lambda x: Mmvn(x,muhist[-1],lamhist[-1],Indicadora0),1,X[Zhist[-1]==update]))
                
    
                exp1=(mu[update]-xi[update])@(tau[update]*SigmainvmNueva[update,:,:])@(mu[update]-xi[update])
                exp2=(mu[update+1]-xi[update+1])@(tau[update+1]*SigmainvmNueva[update+1,:,:])@(mu[update+1]-xi[update+1])
                exp0=(muhist[-1][update]-xihist[-1][update])@(tauhist[-1][update]*Sigmainvm[update,:,:])@(muhist[-1][update]-xihist[-1][update])
                r1=mp.exp(-(exp1-exp2-exp0)/2)
                
                
                r3=len(mu)*mp.fprod((w[update]**niNuevo[update],w[update+1]**niNuevo[update+1]))/(whist[-1][update]*sc.beta(delta, len(muhist[-1])))
                r4=(1/np.pi)*mp.fprod((lam[update]*lam[update+1]/lamhist[-1][update])*np.sqrt((  (lam[update]*linv[update]*2)  )*(  (lam[update+1]*linv[update+1]*2)  )/(  (lamhist[-1][update]*linvhist[-1][update]*2)  )))
                r5=mp.exp(-np.sum(lam[update]*linv[update]+lam[update+1]*linv[update+1]-lamhist[-1][update]*linvhist[-1][update])/2)
                r6=mp.fprod((1/lamhist[-1][update])**(3/2)*(1-ud**2))
                r7=mp.fprod(sp.stats.beta.pdf(np.abs(ud),a=2,b=2))
                
    
                try:
                  num=mp.fprod(np.apply_along_axis(lambda x: Mmvn(x,mu,lam,Indicadora1),1,X[(Z==update)])*w[update]) * mp.fprod(np.apply_along_axis(lambda x: Mmvn(x,mu,lam,Indicadora2),1,X[(Z==update+1)])*w[update+1])
                except:
                  num=0
                try:
                  den=mp.fprod(np.apply_along_axis(lambda x: Mmvn(x,mu,lam,Indicadora1),1,X[(Z==update) | (Z==update+1)])*w[update]+np.apply_along_axis(lambda x: Mmvn(x,mu,lam,Indicadora2),1,X[(Z==update) | (Z==update+1)])*w[update+1])
                  r8=num/den
                except:
                  r8=0
                  
                r9=mp.fprod(sp.stats.norm.pdf(auxXi)*2**(len(auxXi)))*sp.stats.expon.pdf(auxTau,scale=escala)*mp.fprod(sp.stats.expon.pdf( auxlinv,scale=escala))*mp.fprod(1/auxlinv**2)*1/auxTau**2
                # r9=mp.fprod(sp.stats.norm.pdf(auxXi)*2**(len(auxXi)+len(auxlinv)+1)* (auxTau<1)*(auxTau>0)*(all (auxlinv<0.01)) * (all (auxlinv>0))*100) #mas uno por la tau
                
              
   
                
                            
                
                


                R=r0*r1*r2*r3*r4*r5*r6*r7*r8*r9
            except :
                R=0
            

            #Deje fijo rk como 1 entonces el cociente del penultimo renglon es 1/sqrt(pi)


            # x = np.linspace(-5,10,50)
            # y = np.linspace(-5,10,50)
            # Xg,Yg = np.meshgrid(x,y)

            # pdf = np.zeros(Xg.shape)
            # for i in range(Xg.shape[0]):
            #     for j in range(Xg.shape[1]):
            #         Obs=np.array((Xg[i,j], Yg[i,j]))
            #         pdf[i,j] = Mmvn(Obs,mu,lam,w)
            #         # pdf[i,j] = Mmvn(Obs,muhist[-1],lamhist[-1],whist[-1])

            # plt.plot()
            # plt.contourf(Xg, Yg, np.exp(pdf), cmap='viridis',levels=1000)
            # plt.scatter(mu[:,0],mu[:,1])
            # plt.scatter(X[:,0],X[:,1],alpha=0.1)
            # plt.title(R)
            # plt.colorbar()
            # plt.show()
            if np.random.uniform()<np.min((R,1)):
                print("split",R,M)
                reorder=mu[:,0].argsort()
                mu=mu[reorder,:]
                lam=lam[reorder,:]
                w=w[reorder]
                xi=xi[reorder,:]
                tau=tau[reorder]
                linv=linv[reorder,:]

                cont=0
                for p in reorder:
                    Z[Z==p]=cont+M
                    cont+=1

                Z=Z-M

                whist.append(w)
                muhist.append(mu)
                lamhist.append(lam)
                xihist.append(xi)
                tauhist.append(tau)
                linvhist.append(linv)
                Zhist.append(Z)

        ################################################################################################################################################

        

        ####################################falta xi tau linv Z

        #####combine
        if split==0 and M>1:

            Sigmainvm=np.copy(Sigmam)
            for i in range(M):
                Sigmainvm[i,:,:]=np.linalg.inv(Sigmam[i,:,:])


            if M>2:
                ind1=np.random.choice(np.arange(M-2)+1)
                ind2=ind1+np.random.choice((-1,1))
            else :
                ind1=0
                ind2=1


            ind1,ind2=np.sort((ind1,ind2))

            waux=w[ind1]+w[ind2]
            ##
            alpha=w[ind1]/waux
            ##
            
            muaux=(mu[ind1]*w[ind1]+mu[ind2]*w[ind2])/waux
            
            lamaux=1/((w[ind1]*1/lam[ind1]+w[ind2]*1/lam[ind2])/waux+w[ind1]*w[ind2]/waux**2*(mu[ind1]-mu[ind2])**2/np.linalg.norm(mu[ind1]-mu[ind2])**2)

            ud=np.linalg.solve(np.vstack((np.sqrt(1/lamaux[0])*A[:,0],np.sqrt(1/lamaux[1])*A[:,1])),(mu[ind2]-mu[ind1])/(2*np.sqrt(w[ind1]/w[ind2])+1/np.sqrt(w[ind1]/w[ind2])))
            
            betad=(1/lamaux)/(1-ud**2)*w[ind1]/waux*(lam[ind1])
            
            #lam[ind1]-lamaux/(w[ind1]/waux)*betad*(1-ud**2)
            

            auxXi=(xi[ind1]-xi[ind2])/2
            auxTau=tau[ind2]/tau[ind1]
            auxlinv=linv[ind2]/linv[ind1]

            # auxTau=(tau[ind1]-tau[ind2])/2
            # auxlinv=(linv[ind1]-linv[ind2])/2

            
            
            
            
            # mu[ind1]*w[ind1]+mu[ind2]*w[ind2]-muaux*waux
            

            


            mu=np.delete(mu, ind2, 0)
            mu[ind1]=muaux

            lam=np.delete(lam, ind2, 0)
            lam[ind1]=lamaux


            w=np.delete(w,ind2)
            w[ind1]=waux
            
            
            xi[ind1]=(xi[ind1]+xi[ind2])/2
            xi=np.vstack((xi[:ind1+1,:] ,xi[ind2+1:,:]))

            tau[ind1]=(tau[ind1]+tau[ind2])/2
            tau=np.hstack((tau[:ind1+1] ,tau[ind2+1:]))

            linv[ind1]=(linv[ind1]+linv[ind2])/2
            linv=np.vstack((linv[:ind1+1,:] ,linv[ind2+1:,:]))




            ##########################

            ###

            # Z[ (Z==ind1) + (Z==ind2)]=ind1
            # Z[Z>ind2]=Z[Z>ind2]-1
            ###
            M=len(mu)

            SigmamNueva=np.zeros((M,2,2))

            for i in range(M):
                SigmamNueva[i,:,:]=A@np.diag(1/lam[i])@A.T

            detSigmaNueva=np.zeros(M)

            for i in range(M):
                detSigmaNueva[i]=np.linalg.det(SigmamNueva[i,:,:])


            SigmainvmNueva=np.copy(SigmamNueva)
            for i in range(M):
                SigmainvmNueva[i,:,:]=np.linalg.inv(SigmamNueva[i,:,:])

            
            Z[(Z==ind1)|(Z==ind2)]=ind1
            Z[Z>ind1]=Z[Z>ind1]-1            
            
            
            
            # niNuevo=np.zeros(M)
            # for i in range(M):
            #     niNuevo[i]=np.sum(Z==i)
            # print(niNuevo)

            ###


            try :
                r0=priorXi.pdf(x=xihist[-1][update])*priorXi.pdf(x=xihist[-1][update+1])/priorXi.pdf(x=xi[update])
                r0*=mp.fprod((priorTau.pdf(tauhist[-1][update]),priorTau.pdf(tauhist[-1][update+1])))/priorTau.pdf(x=tau[update])
                r0*=np.prod(priorlinv.pdf(linvhist[-1][update]))*np.prod(priorlinv.pdf(linvhist[-1][update+1]))/np.prod(priorlinv.pdf(linv[update]))
                
                Indicadora0=np.zeros(M)
                Indicadora1=np.zeros(M+1)
                Indicadora2=np.zeros(M+1)
                Indicadora0[ind1] = 1
                Indicadora1[ind1] = 1
                Indicadora2[ind2] =1
                
                            
                r2=mp.fprod(np.apply_along_axis(lambda x: Mmvn(x,muhist[-1],lamhist[-1],Indicadora1),1,X[Zhist[-1]==ind1]))*mp.fprod(np.apply_along_axis(lambda x: Mmvn(x,muhist[-1],lamhist[-1],Indicadora2),1,X[Zhist[-1]==ind2]))/mp.fprod(np.apply_along_axis(lambda x: Mmvn(x,mu,lam,Indicadora0),1,X[Z==ind1]))
                
                update=np.copy(ind1)
                exp1=(muhist[-1][update]-xihist[-1][update])@(tauhist[-1][update]*Sigmainvm[update,:,:])@(muhist[-1][update]-xihist[-1][update])
                exp2=(muhist[-1][update+1]-xihist[-1][update+1])@(tauhist[-1][update+1]*Sigmainvm[update+1,:,:])@(muhist[-1][update+1]-xihist[-1][update+1])
                exp0=(mu[update]-xi[update])@(tau[update]*SigmainvmNueva[update,:,:])@(mu[update]-xi[update])
                r1=mp.exp(-(exp1-exp2-exp0)/2)
                
                
                r3=(len(mu)+1)*mp.fprod((whist[-1][update]**np.sum(Zhist[-1]==update),whist[-1][update+1]**np.sum(Zhist[-1]==update+1)))/(w[update]*sc.beta(delta, len(mu)))
                r4=(1/np.pi)*mp.fprod((lamhist[-1][update]*lamhist[-1][update+1]/lam[update])*np.sqrt((  (lamhist[-1][update]*linvhist[-1][update]*2)  )*(  (lamhist[-1][update+1]*linvhist[-1][update+1]*2)  )/(  (lam[update]*linv[update]*2)  )))
                r5=mp.exp(-np.sum(lamhist[-1][update]*linvhist[-1][update]+lamhist[-1][update+1]*linvhist[-1][update+1]-lam[update]*linv[update])/2)
                r6=mp.fprod((1/lamhist[-1][update])**(3/2)*(1-ud**2))
                r7=mp.fprod(sp.stats.beta.pdf(np.abs(ud),a=2,b=2))
                
    
                try:
                  num=mp.fprod(np.apply_along_axis(lambda x: Mmvn(x,muhist[-1],lamhist[-1],Indicadora1),1,X[(Zhist[-1]==update)])*whist[-1][update]) * mp.fprod(np.apply_along_axis(lambda x: Mmvn(x,muhist[-1],lamhist[-1],Indicadora2),1,X[(Zhist[-1]==update+1)])*whist[-1][update+1])
                except:
                  num=0
                try:
                  den=mp.fprod(np.apply_along_axis(lambda x: Mmvn(x,muhist[-1],lamhist[-1],Indicadora1),1,X[(Zhist[-1]==update) | (Zhist[-1]==update+1)])*whist[-1][update]+np.apply_along_axis(lambda x: Mmvn(x,mu,lam,Indicadora2),1,X[(Zhist[-1]==update) | (Zhist[-1]==update+1)])*whist[-1][update+1])
                  r8=num/den
                except:
                  r8=0
                 
                 
                # r9=mp.fprod(sp.stats.norm.pdf(auxXi)*2**(len(auxXi)+len(auxlinv)+1)* (auxTau<1)*(auxTau>0)*(all (auxlinv<0.01)) * (all (auxlinv>0))*100) #mas uno por la tau
                r9=mp.fprod(sp.stats.norm.pdf(auxXi)*2**(len(auxXi)))*sp.stats.expon.pdf(auxTau,scale=escala)*mp.fprod(sp.stats.expon.pdf( auxlinv,scale=escala))*mp.fprod(1/auxlinv**2)*1/auxTau**2
                


                R=1/(r0*r1*r2*r3*r4*r5*r6*r7*r8*r9)
            except :
                R=0
                
                
            #Deje fijo rk como 1 entonces el cociente del penultimo renglon es 1/sqrt(pi)


            # x = np.linspace(-5,10,50)
            # y = np.linspace(-5,10,50)
            # Xg,Yg = np.meshgrid(x,y)

            # pdf = np.zeros(Xg.shape)
            # for i in range(Xg.shape[0]):
            #     for j in range(Xg.shape[1]):
            #         Obs=np.array((Xg[i,j], Yg[i,j]))
            #         pdf[i,j] = Mmvn(Obs,mu,lam,w)
            #         # pdf[i,j] = Mmvn(Obs,muhist[-1],lamhist[-1],whist[-1])

            # plt.plot()
            # plt.contourf(Xg, Yg, np.exp(pdf), cmap='viridis',levels=1000)
            # plt.scatter(mu[:,0],mu[:,1])
            # plt.scatter(X[:,0],X[:,1],alpha=0.1)
            # plt.title(R)
            # plt.colorbar()
            # plt.show()
            if np.random.uniform()<np.min((R,1)):
                print("combine",R,M)
                reorder=mu[:,0].argsort()
                mu=mu[reorder,:]
                lam=lam[reorder,:]
                w=w[reorder]
                xi=xi[reorder,:]
                tau=tau[reorder]
                linv=linv[reorder,:]

                cont=0
                for p in reorder:
                    Z[Z==p]=cont+M
                    cont+=1

                Z=Z-M

                whist.append(w)
                muhist.append(mu)
                lamhist.append(lam)
                xihist.append(xi)
                tauhist.append(tau)
                linvhist.append(linv)
                Zhist.append(Z)



########################

    if len(muhist[-1])!=Mhist[-1] :
        Mhist.append(len(muhist[-1]))
        print(linv,"\n\n",lamhist[-2])
    else :
        Mhist.append(Mhist[-1])
    
    #print(Mhist[-1])

    if q%100==0:



        x = np.linspace(np.min(X[:,0]),np.max(X[:,0]),50)
        y = np.linspace(np.min(X[:,1]),np.max(X[:,1]),50)
        Xg,Yg = np.meshgrid(x,y)

        pdf = np.zeros(Xg.shape)
        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[1]):
                Obs=np.array((Xg[i,j], Yg[i,j]))
                pdf[i,j] = Mmvn(Obs,muhist[-1],lamhist[-1],whist[-1])

        plt.plot()
        plt.contourf(Xg, Yg, pdf, cmap='viridis',levels=1000)
        plt.scatter(X[:,0],X[:,1],alpha=0.1)
        plt.scatter(muhist[-1][:,0],muhist[-1][:,1],c="red",s=50,alpha=0.5)
        plt.title(Mhist[-1])
        plt.colorbar()
        plt.show()







plt.plot(np.arange(len(Mhist)),Mhist)
plt.show()
        ####################################falta xi tau linv Z


    # #####Falta Nacimiento muerte

    # if step<4:
    #     whist.append(w)
    #     muhist.append(mu)
    #     lamhist.append(lam)
    #     xihist.append(xi)
    #     tauhist.append(tau)
    #     linvhist.append(linv)
    #     Zhist.append(Z)

    # if step==4:




    # if q%100:



    #     x = np.linspace(-5,10,50)
    #     y = np.linspace(-5,10,50)
    #     Xg,Yg = np.meshgrid(x,y)

    #     pdf = np.zeros(Xg.shape)
    #     for i in range(Xg.shape[0]):
    #         for j in range(Xg.shape[1]):
    #             Obs=np.array((Xg[i,j], Yg[i,j]))
    #             pdf[i,j] = Mmvn(Obs)

    #     plt.plot()
    #     plt.contourf(Xg, Yg, np.exp(pdf), cmap='viridis',levels=1000)
    #     plt.scatter(mu[:,0],mu[:,1])
    #     plt.scatter(X[:,0],X[:,1],alpha=0.1)
    #     plt.colorbar()
    #     plt.show()















# UTIL PARA PRUEBAS

# ############################################################################################################################################
#             w=whist[-1]
#             mu=muhist[-1]
#             lam=lamhist[-1]
#             xi=xihist[-1]
#             tau=tauhist[-1]
#             linv=linvhist[-1]
#             Z=Zhist[-1]

#             step=np.random.choice(np.arange(5))


#             XZ=(pd.DataFrame((X[:,0],X[:,1],np.copy(Z)))).transpose()
#             XZ.columns=("x","y","g")


#             ############Actualizar
#             M=len(mu)
#             Sigmam=np.zeros((M,2,2))

#             for i in range(M):
#                 Sigmam[i,:,:]=A@np.diag(1/lam[i])@A.T

#             detSigma=np.zeros(M)

#             for i in range(M):
#                 detSigma[i]=np.linalg.det(Sigmam[i,:,:])


#             ni=np.zeros(M)
#             for i in range(M):
#                 ni[i]=np.sum(XZ["g"]==i)
#             print(ni)
# ############################################################################################################################################




















