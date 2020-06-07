import numpy as np
import matplotlib.pyplot as plt
import random
import math
import powerlaw             # generador power law
N=50
medidas=10000

# FUNCIONES
def kmean(m):
    '''
    Función que devuelve:  <k> ,  <k>^2  ,  <k^2>  ,  <k^2knn>  y  <k^3>
    '''
    km=(np.count_nonzero(m)/float(N))
    k=km**2.0
    k2=0.0
    k3=0.0
    knn=np.zeros(N, dtype=float)
    for i in range(N):
        k2+=float(sum(m[:,i]))**2.0
        k3+=float(sum(m[:,i]))**3.0
        for j in range(N):
            if m[j,i]==1:
                knn[i]+=float(sum(m[:,j]))
        knn[i]/=float(sum(m[:,i]))
        knn[i]*=float(sum(m[:,i]))**2.0
    k2/=float(N)
    k3/=float(N)
    knnm=sum(knn)/float(N)
    return (km,k,k2,knnm,k3)

def nest(m, a, b):
    '''
    Función que devuelve el nestedness entre dos elementos
    '''
    eta=0.0
    for i in range(N):
        eta+=m[a,i]*m[b,i]
    eta/=(float(sum(m[:,a]))*float(sum(m[:,b])))
    return eta
        
def nestotal(m):
    '''
    Función que devuelve el nestedness medio de la red
    '''
    etam=0.0
    for i in range(N):
        for j in range(N):
            etam+=nest(m,i,j)
    etam/=float(N)
    return etam

def normnest(m):
    '''
    Función que devuelve el nestedness normalizado (6)
    '''
    etan=0.0
    for i in range(N):
        for j in range(N):
            etan+=nest(m,i,j)         # igual
    etan*=(kmean(m)[1]/(kmean(m)[2]*float(N)))
    return etan

def pearson(m):
    '''
    Función que devuelve el coef de pearson
    '''
    rr=kmean(m)
    r=(rr[0]*rr[3]-rr[2]**2)/(rr[0]*rr[4]-rr[2]**2)
    return r

# FUNCIONES DE GENERACIÓN DE MATRICES (hay 3 pero se pueden juntar en 1)
def gaus(L, mu, sigma):
    '''
    Función que crea una matriz de ady gaussiana mediante configuration model
    '''
    v1=[]
    v2=[]
    count=0
    m=np.zeros([L,L], dtype=int)
    # creo las listas con sublistas con un numero de elementos igual a cada grado
    for i in range(L):
        # aqui hay una cuestion: la distribucion de grados de un conjunto
        # puede ser IGUAL o DEL MISMO TIPO que la del otro
        # Por simplicidad lo pongo igual
        v1.append([])
        v2.append([])
        nmb=0
        while nmb<1:
            nmb=math.floor(random.gauss(mu,sigma))
        for j in range(nmb):
            ap=1
            v1[i].insert(i,ap)
            v2[i].insert(i,ap)
            count+=1

    # ahora tengo una lista con N sublistas, cada una con k unos distribuidos gaussianamente
    # para generar la matriz, escojo dos elementos de dos en dos de v1 y v2 y los elimino, y 
    # la componente de la matrix correspondiente la hago 1
   
    for it in range(count):
        rn1=random.randint(0,count-it-1)
        rn2=random.randint(0,count-it-1)
        count1=0 # indice total del elemento dentro de toda la lista
        ind1=0 # indice de sublista
        count2=0
        ind2=0 
        for i in v1:
            for j in i:
                if count1==rn1:
                    i.remove(1)
                    coor1=ind1
                count1+=1
            ind1+=1
        for i in v2:
            for j in i:
                if count2==rn2:
                    i.remove(1)
                    coor2=ind2
                count2+=1
            ind2+=1

        m[coor1,coor2]=1
    return m

def pois(L, mu):
    '''
    Función que crea una matriz de ady poissoniana mediante configuration model
    '''
    v1=[]
    v2=[]
    count=0
    m=np.zeros([L,L], dtype=int)
    for i in range(L):
        v1.append([])
        v2.append([])
        nmb=0
        while nmb<1:
            nmb=math.floor(np.random.poisson(mu))
        for j in range(nmb):
            ap=1
            v1[i].insert(i,ap)
            v2[i].insert(i,ap)
            count+=1

    for it in range(count):
        rn1=random.randint(0,count-it-1)
        rn2=random.randint(0,count-it-1)
        count1=0 
        ind1=0 
        count2=0
        ind2=0 
        for i in v1:
            for j in i:
                if count1==rn1:
                    i.remove(1)
                    coor1=ind1
                count1+=1
            ind1+=1
        for i in v2:
            for j in i:
                if count2==rn2:
                    i.remove(1)
                    coor2=ind2
                count2+=1
            ind2+=1

        m[coor1,coor2]=1
    return m

def scalef(L, gamma):
    '''
    Función que crea una matriz de ady scale free mediante configuration model
    '''
    v1=[]
    v2=[]
    count=0
    m=np.zeros([L,L], dtype=int)
    for i in range(L):
        v1.append([])
        v2.append([])
        num=0
        while num<1 or num>N:
            nmb=powerlaw.Power_Law(xmin=1, parameters=[gamma]).generate_random(1)
            num=nmb[0]
        for j in range(math.floor(num)):
            ap=1
            v1[i].insert(i,ap)
            v2[i].insert(i,ap)
            count+=1

    for it in range(count):
        rn1=random.randint(0,count-it-1)
        rn2=random.randint(0,count-it-1)
        count1=0 
        ind1=0 
        count2=0
        ind2=0 
        for i in v1:
            for j in i:
                if count1==rn1:
                    i.remove(1)
                    coor1=ind1
                count1+=1
            ind1+=1
        for i in v2:
            for j in i:
                if count2==rn2:
                    i.remove(1)
                    coor2=ind2
                count2+=1
            ind2+=1

        m[coor1,coor2]=1
    return m

# PROCESO DE MUESTREO
muestra=[]
for i in tqdm(range(medidas)):
    mg=pois(N, 10)
    pear=pearson(mg)
    muestra.append(pear)

#MUESTRO RESULTADOS
pesos=np.ones_like(muestra)/float(len(muestra))
plt.hist(muestra, bins=60, color='red')
plt.xlim(-0.7,0.3)
plt.ylim(0,3500)
plt.xlabel("r")
plt.show()

# OTROS RESULTADOS 
# cosapo=scalef(N, 2.25)
# histopo=[]
# for i in range(N):
#     histopo.append(sum(cosapo[:,i]))
# plt.hist(histopo,width=1, bins=range(2*int(min(histopo))-1, 2*int(max(histopo)+1))) 
# plt.xlabel("k")
# plt.ylabel("n")
# plt.yscale('log')
# plt.show()

# cosa=gaus(N, 10, 2)
# histo=[]
# for i in range(N):
#     histo.append(sum(cosa[:,i]))
# plt.hist(histo, bins=25) 
# plt.title("dist de grados gauss") 
# plt.show()

# print("numero total de uniones: %s" % (np.count_nonzero(cosa)))
# print(cosa)
# print(kmean(cosa))
# print("pearson: %s" % (pearson(cosa)))
# print("nestotal: %s" % (nestotal(cosa)))
# print("normnest: %s" % (normnest(cosa)))
# cosilla=nest(cosa,0,1)
# print("nest 0,1: %s" % (cosilla))
