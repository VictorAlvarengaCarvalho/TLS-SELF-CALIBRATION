# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:45:05 2020

@author: victo
"""
# -------------------------- Bibliotecas --------------------------
from tkinter import *
from tkinter import filedialog
import sympy as sym
import pandas as pd
import scipy as sc
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn
import numpy as np
from numpy.linalg import inv
# -------------------------- Entradas --------------------------
DIR1 = "Model/" #Diretorio das observações
DIR2 = "Model/Results/" #Diretorio salvamento dos arquivos
DIR3 = "" #Diretorio dos parametros

# -- Atribuindo os dados das tabelas a DataFrame -- 
observacoes = pd.read_csv(DIR1+"observations.csv",sep = ";")
vetor = pd.read_csv(DIR1+"plane.csv",sep = ";",index_col=0)
parametros = pd.read_csv(DIR3+"parameters.csv",sep =";",index_col=0)

# -- Preparar o  DataFrame -- 
estacao = [0,7] # Estações utilizadas no processo
plano = [1,2,3,4,5,6] # Planos utilizados
teste = str(7)
limite = 4
Vcorrecao = 1e-06
Lang = 1e-06
Lcoord = 1e-06
Laps = 1e-06
parametros= parametros.loc[estacao] # Limpar os parametros
vetor = vetor.loc[plano] # Limpara os vetores
observacoes['PlanoF'] = observacoes.iloc[:,1] # Auxiliar para filtrar os planos
observacoes = observacoes.set_index("PlanoF")
observacoes = observacoes.loc[plano] 
observacoes = observacoes.set_index("Estacao")
observacoes = observacoes.loc[estacao] # Filtado de planos e estações
# -- Constate do motelo --
constante = sym.ones(1,len(plano)) # vetor de constante
# -- Validar a entrada de dados --
n = observacoes.shape[0]*3 # Numero de observacoes
u = len(estacao)*6+3 # Numero de parametros livres (nao conta as restrições)
c = observacoes.shape[0]*3 # Numero de equações de condição 
r = n-u # Grau de liberdade
if r < 0  : # Testa o grau de liberdade
    print('Vareiaveis não antendem as condiçoes minimas para o ajustamento') # Mensagem par ao usuario
    exit()
for i in estacao: # Contar o numero de pontos por plano
    observacoesest = observacoes.loc[i]
    counts = observacoesest ['Plano'].value_counts()
    if min(counts) < 3: # Testa se há mais de 3 observacoes por plano
        print('Número de observacoes insuficientes por plano')
        exit()
# -- Variaveis estatisticas --
varp =sym.Matrix([1]) # Sigma a priori
Q = 1e-9*sym.eye(observacoes.shape[0]*3) # Observações  5mm
QC = 1e-9*sym.eye(observacoes.shape[0]) # Observações de restrição 0.005mm
QP = 1e-9*sym.eye(len(plano)*4) # Parametros do plano 0.0001mm
QA = 1e-11*sym.eye(3) # Angulo de orintação Espaço objeto 0.1 s
QD = 1e-9*sym.eye(3) # Posição da Estação 0.1mm
QADa = 1e+10 # Parametros Adicionais Constante aditiva
QADb0 = 1e+10 # Parametros Adicionais correções angulares
QADb1 = 1e+10
QADc = 1e+10
# QADb0 = 1e-06 # Parametros Adicionais correções angulares
# QADb1 = 1e-05
# QADc = 1e-05
QAD = sym.diag(QADa,QADb0)
QAD = sym.diag(QAD,QADb1)
QAD = sym.diag(QAD,QADc)
W = Q.inv()
QXX = QP # Auxiliar

for i in range(len(estacao)): # Concatenando as matrizes conforme o numero de estaçoes 
    QXX = sym.diag(QXX,QA)
    QXX = sym.diag(QXX,QD)
QXX = sym.diag(QXX,QAD)
WXX = QXX.inv()
# -- Vetores iniciais do ajustamento --
V = sym.zeros(1,observacoes.shape[0]*3).T# Residuo observacoes 
VC = sym.zeros(1,observacoes.shape[0]).T # Residuo Restrição
CONS = sym.ones(1,len(plano))# Vetor da contante relativa aos planos
# -------------------------- Variaveis do Modelo --------------------------
# Matrizes fundamentais para o processo
# X = [a,b,c,d,omega,kappa,phi,X,Y,Z,a0,b0,c0] 
# LB = [x,y,z]
omega, kappa, phi = sym.symbols('omega,kappa,phi') #Rotação
x, y, z = sym.symbols('x, y, z') #Coordenada cada ponto no espaço imagem
X, Y, Z = sym.symbols('X, Y, Z') #Centragem da estação
a0, b0, b1, c0 = sym.symbols('a0, b0, b1, c0') # Parametros de calibração
a, b, c, d, cons= sym.symbols('a, b, c, d,cons') #Variaveis do plano
# -------------------------- Vetores corrigidos --------------------------
# Atualizados a cada iteração
obscorrigido = observacoes
vcorrigido = vetor
parcorrugido = parametros
# -------------------------- Modelo Matemático --------------------------
def ModeloFuncional():
    #Declarando as variaveis simbolicas
    omega, kappa, phi = sym.symbols('omega,kappa,phi') #Rotação
    x, y, z = sym.symbols('x, y, z') #Coordenada cada ponto no espaço imagem
    X, Y, Z = sym.symbols('X, Y, Z') #Centragem da estação
    a0, b0, b1, c0 = sym.symbols('a0, b0, b1, c0') # Parametros de calibração
    a, b, c, d, cons= sym.symbols('a, b, c, d,cons') #Variaveis do plano
    #Matriz de rotação
    wi =sym.Matrix ([(1, 0, 0), (0, sym.cos(omega), -sym.sin(omega)), (0, sym.sin(omega), sym.cos(omega))])
    phii = sym.Matrix ([(sym.cos(phi), 0, sym.sin(phi)), (0, 1, 0), (-sym.sin(phi), 0, sym.cos(phi))])
    ki = sym.Matrix([(sym.cos(kappa), -sym.sin(kappa), 0), (sym.sin(kappa), sym.cos(kappa), 0), (0, 0, 1)])
    mj = ki * phii * wi #Matriz de rotação
    #Decomposição das coordenadas cartesianas em Coordenadas polares
    alpha = sym.atan2(z,(x**2 + y**2)**(0.5))
    teta = sym.atan2(y,x)
    ro = (x**2 + y**2 + z**2)**(0.5)
    dalpha = c0
    dteta = b0*sym.sec(alpha)+b1*sym.tan(alpha)
    dro = a0
    TT = sym.Matrix([(ro - dro) * sym.cos(alpha - dalpha) * sym.cos(teta - dteta),
                    (ro - dro) * sym.cos(alpha - dalpha) * sym.sin(teta - dteta),
                    (ro - dro) * sym.sin(alpha - dalpha)]) #Matriz de coordenadas dos pontos adicionado os APs 
    Vplano = sym.Matrix ([(a, b, c)]) # Parametros de definição do plano
    #Equações f - função que descreve o problema, g - função de restrição 
    f = sym.Matrix([(Vplano*(mj*TT+sym.Matrix([X, Y, Z])) - sym.Matrix([d]))]) # Modelo matemático
    g = sym.Matrix([(a**2+b**2+c**2)-cons]) # Função de restrição
    return f,g
# -------------------------- Linearização --------------------------
def derivadas():
    f,g = ModeloFuncional()
    #Variaveis
    omega, kappa, phi = sym.symbols('omega,kappa,phi') 
    x, y, z = sym.symbols('x, y, z')
    X, Y, Z = sym.symbols('X, Y, Z')
    a0, b0, b1, c0 = sym.symbols('a0, b0, b1, c0') # Parametros de calibração
    a, b, c, d,cons = sym.symbols('a, b, c, d,cons')
    #Vetor das observacoes
    LB = sym.Matrix([(x,y,z)])
    #Vetor dos parametros subdividido (Possibilita montar a matriz A)
    PLANO = sym.Matrix([(a,b,c,d)])
    ESTACAO = sym.Matrix([(omega, kappa, phi ,X, Y, Z)])
    APS = sym.Matrix([(a0, b0, b1, c0)])
    #Vetor das restrições
    LC = sym.Matrix([(cons)])
    #Matrizes das derivadas em relação a cada conjunto de variaveis
    BP = f.jacobian(PLANO)
    BE = f.jacobian(ESTACAO)
    BAD = f.jacobian(APS)
    CP = g.jacobian(PLANO)
    CE = g.jacobian(ESTACAO)
    CAD = g.jacobian(APS)
    A = f.jacobian(LB)
    AC = g.jacobian(LC)
    return BP,BE,BAD,CP,CE,CAD,A,AC
# -------------------------- Substituição de vetores --------------------------
def Aplicar (BP,BE,BAD,CP,CE,CAD,A,AC,X0,LB,CONS):
    #Variaveis
    omega, kappa, phi = sym.symbols('omega,kappa,phi')
    x, y, z = sym.symbols('x, y, z')
    X, Y, Z = sym.symbols('X, Y, Z')
    a0, b0, b1, c0 = sym.symbols('a0, b0, b1, c0') # Parametros de calibração
    a, b, c, d,cons = sym.symbols('a, b, c, d,cons')
    #Matrizes do processo
    BP,BE,BAD,CP,CE,CAD,A,AC = derivadas()
    f,g = ModeloFuncional()
    #Subistituição para o modelo numérico
    BP = BP.subs([(x,LB[0,0]), (y,LB[0,1]),(z,LB[0,2]),
                  (a,X0[0,0]), (b,X0[0,1]), (c,X0[0,2]), (d,X0[0,3]),
                  (omega,X0[0,4]), (kappa,X0[0,5]), (phi,X0[0,6]), 
                  (X,X0[0,7]), (Y,X0[0,8]),(Z,X0[0,9]),
                  (a0,X0[0,10]), (b0,X0[0,11]),(b1,X0[0,12]), (c0,X0[0,13]),(cons,CONS[0,0])])
    BE = BE.subs([(x,LB[0,0]), (y,LB[0,1]),(z,LB[0,2]),
                  (a,X0[0,0]), (b,X0[0,1]), (c,X0[0,2]), (d,X0[0,3]),
                  (omega,X0[0,4]), (kappa,X0[0,5]), (phi,X0[0,6]), 
                  (X,X0[0,7]), (Y,X0[0,8]),(Z,X0[0,9]),
                  (a0,X0[0,10]), (b0,X0[0,11]),(b1,X0[0,12]), (c0,X0[0,13]),(cons,CONS[0,0])])
    BAD = BAD.subs([(x,LB[0,0]), (y,LB[0,1]),(z,LB[0,2]),
                  (a,X0[0,0]), (b,X0[0,1]), (c,X0[0,2]), (d,X0[0,3]),
                  (omega,X0[0,4]), (kappa,X0[0,5]), (phi,X0[0,6]), 
                  (X,X0[0,7]), (Y,X0[0,8]),(Z,X0[0,9]),
                  (a0,X0[0,10]), (b0,X0[0,11]),(b1,X0[0,12]), (c0,X0[0,13]),(cons,CONS[0,0])])
    CP = CP.subs([(x,LB[0,0]), (y,LB[0,1]),(z,LB[0,2]),
                  (a,X0[0,0]), (b,X0[0,1]), (c,X0[0,2]), (d,X0[0,3]),
                  (omega,X0[0,4]), (kappa,X0[0,5]), (phi,X0[0,6]), 
                  (X,X0[0,7]), (Y,X0[0,8]),(Z,X0[0,9]),
                  (a0,X0[0,10]), (b0,X0[0,11]),(b1,X0[0,12]), (c0,X0[0,13]),(cons,CONS[0,0])])
    CE = CE.subs([(x,LB[0,0]), (y,LB[0,1]),(z,LB[0,2]),
                  (a,X0[0,0]), (b,X0[0,1]), (c,X0[0,2]), (d,X0[0,3]),
                  (omega,X0[0,4]), (kappa,X0[0,5]), (phi,X0[0,6]), 
                  (X,X0[0,7]), (Y,X0[0,8]),(Z,X0[0,9]),
                  (a0,X0[0,10]), (b0,X0[0,11]),(b1,X0[0,12]), (c0,X0[0,13]),(cons,CONS[0,0])])
    CAD = CAD.subs([(x,LB[0,0]), (y,LB[0,1]),(z,LB[0,2]),
                  (a,X0[0,0]), (b,X0[0,1]), (c,X0[0,2]), (d,X0[0,3]),
                  (omega,X0[0,4]), (kappa,X0[0,5]), (phi,X0[0,6]), 
                  (X,X0[0,7]), (Y,X0[0,8]),(Z,X0[0,9]),
                  (a0,X0[0,10]), (b0,X0[0,11]),(b1,X0[0,12]), (c0,X0[0,13]),(cons,CONS[0,0])])
    A = A.subs([(x,LB[0,0]), (y,LB[0,1]),(z,LB[0,2]),
                  (a,X0[0,0]), (b,X0[0,1]), (c,X0[0,2]), (d,X0[0,3]),
                  (omega,X0[0,4]), (kappa,X0[0,5]), (phi,X0[0,6]), 
                  (X,X0[0,7]), (Y,X0[0,8]),(Z,X0[0,9]),
                  (a0,X0[0,10]), (b0,X0[0,11]),(b1,X0[0,12]), (c0,X0[0,13]),(cons,CONS[0,0])])
    AC = AC.subs([(cons,CONS[0,0])])
    f = f.subs([(x,LB[0,0]), (y,LB[0,1]),(z,LB[0,2]),
                  (a,X0[0,0]), (b,X0[0,1]), (c,X0[0,2]), (d,X0[0,3]),
                  (omega,X0[0,4]), (kappa,X0[0,5]), (phi,X0[0,6]), 
                  (X,X0[0,7]), (Y,X0[0,8]),(Z,X0[0,9]),
                  (a0,X0[0,10]), (b0,X0[0,11]),(b1,X0[0,12]), (c0,X0[0,13]),(cons,CONS[0,0])])
    g = g.subs([(x,LB[0,0]), (y,LB[0,1]),(z,LB[0,2]),
                  (a,X0[0,0]), (b,X0[0,1]), (c,X0[0,2]), (d,X0[0,3]),
                  (omega,X0[0,4]), (kappa,X0[0,5]), (phi,X0[0,6]), 
                  (X,X0[0,7]), (Y,X0[0,8]),(Z,X0[0,9]),
                  (a0,X0[0,10]), (b0,X0[0,11]),(b1,X0[0,12]), (c0,X0[0,13]),(cons,CONS[0,0])])
    return BP,BE,BAD,CP,CE,CAD,A,AC,f,g
# -------------------------- Modelo direto --------------------------
def ModeloDireto(lbAJ,X0):
    # -- Matriz de rotação --
    wi =sym.Matrix ([(1, 0, 0), (0, sym.cos(omega), -sym.sin(omega)), (0, sym.sin(omega), sym.cos(omega))])
    phii = sym.Matrix ([(sym.cos(phi), 0, sym.sin(phi)), (0, 1, 0), (-sym.sin(phi), 0, sym.cos(phi))])
    ki = sym.Matrix([(sym.cos(kappa), -sym.sin(kappa), 0), (sym.sin(kappa), sym.cos(kappa), 0), (0, 0, 1)])
    R = ki * phii * wi
    #Decomposição das coordenadas cartesianas em Coordenadas polares
    alpha = sym.atan2(z,(x**2 + y**2)**(0.5)) 
    teta = sym.atan2(y,x)  
    ro = (x**2 + y**2 + z**2)**(0.5)
    dalpha = c0
    dteta = b0*sym.sec(alpha)+b1*sym.tan(alpha)
    dro = a0
    # -- Matriz de coordenadas dos pontos adicionado os APs -- 
    TT = sym.Matrix([(ro - dro) * sym.cos(alpha - dalpha) * sym.cos(teta - dteta),
                    (ro - dro) * sym.cos(alpha - dalpha) * sym.sin(teta - dteta),
                    (ro - dro) * sym.sin(alpha - dalpha)])
    # -- Aplicas as variaveis --
    mj = R.subs([(omega,X0[0,4]), (kappa,X0[0,5]), (phi,X0[0,6])])
    coordenadasAj = TT.subs([(x,lbAJ[0,0]), (y,lbAJ[0,1]),(z,lbAJ[0,2]),(a0,X0[0,10]), (b0,X0[0,11]), (b1,X0[0,12]), (c0,X0[0,13])])
    Aplicado = mj*coordenadasAj+X0[0,7:10].T
    return Aplicado

# -------------------------- Inicializando das iterações --------------------------
# -- Armazen o valor das derivas (Não é preciso passar pelas iretações) -- 
BP1,BE1,BAD1,CP1,CE1,CAD1,A1,AC1 = derivadas()
MENSAGEM2 = 'Criterio do limite de iterações'
contador = 0
for iteracao in range (0,limite): # Iterações de convergencia do ajustamento
    # -- Inicialixação de vetores do ajustamento -- 
    fx = sym.Matrix([])
    F = sym.Matrix([])
    G = sym.Matrix([])
    BADT = sym.Matrix([])
    CADT = sym.Matrix([])
    BPTTT = sym.Matrix([])
    CPTTT = sym.Matrix([])
    BETT = sym.Matrix([])
    CETT = sym.Matrix([])
    ACT = sym.Matrix([])
    AT = sym.Matrix([])
    F = sym.Matrix([])
    G = sym.Matrix([])
    Ajustado =sym.Matrix([])
    X0 = sym.zeros(1,14)
    XI = sym.zeros(1,14)
    # -------------------------- Percorrer os vetores iniciasi (ponto,plano,estação) --------------------------
    for i in (estacao):# Percorrer todas as estações
        X0[4] = sym.Matrix([(parcorrugido.loc[i,"omega"],parcorrugido.loc[i,"kappa"],parcorrugido.loc[i,"phi"],parcorrugido.loc[i,"X"],parcorrugido.loc[i,"Y"],parcorrugido.loc[i,"Z"],parcorrugido.loc[i,"a0"],parcorrugido.loc[i,"b0"],parcorrugido.loc[i,"b1"],parcorrugido.loc[i,"c0"])]) # X ajustado
        XI[4] = sym.Matrix([(parametros.loc[i,"omega"],parametros.loc[i,"kappa"],parametros.loc[i,"phi"],parametros.loc[i,"X"],parametros.loc[i,"Y"],parametros.loc[i,"Z"],parametros.loc[i,"a0"],parametros.loc[i,"b0"],parametros.loc[i,"b1"],parametros.loc[i,"c0"])]) # X0 por estação
        # -- Matrizes atualizaveis a cada estação percorrida --
        pAJ = obscorrigido.loc[i] # Parametros Ajustado por estação
        BET = sym.Matrix([]) 
        CET = sym.Matrix([])
        BPTT = sym.Matrix([])
        CPTT = sym.Matrix([])
        FAP = sym.Matrix([]) 
        for j in (plano):# Percorrer os planos por estação 
            X0[0] = sym.Matrix([(vcorrigido.loc[j,'Nx'],vcorrigido.loc[j,'Ny'],vcorrigido.loc[j,'Nz'],vcorrigido.loc[j,'d'])])# Adiciona os parametros do plano
            XI[0] = sym.Matrix([(vetor.loc[j,'Nx'],vetor.loc[j,'Ny'],vetor.loc[j,'Nz'],vetor.loc[j,'d'])])
            pontosAJ = pAJ.loc[(pAJ['Plano'])==j]# Pontos de cada plano Ajustado
            BPT = sym.Matrix([])
            CPT = sym.Matrix([])
            for k in range (0, pontosAJ.shape[0]):
                lbAJ = sym.Matrix([(pontosAJ.iloc[k,1],pontosAJ.iloc[k,2],pontosAJ.iloc[k,3])])
                BP,BE,BAD,CP,CE,CAD,A,AC,f,g = Aplicar(BP1,BE1,BAD1,CP1,CE1,CAD1,A1,AC1,X0,lbAJ,CONS)
                F = F.col_join(f)
                G = G.col_join(g)
                # -- Processo de indexação para o ajustamento em bloco -- 
                BPT = BPT.col_join(BP)
                BET = BET.col_join(BE)
                BADT = BADT.col_join(BAD)
                CPT = CPT.col_join(CP)
                CET = CET.col_join(CE)
                CADT = CADT.col_join(CAD)
                AT = sym.diag(AT,A)
                ACT = sym.diag(ACT,AC)
                # -- Aplicando o modelo aos pontos da iteração anterios para obter os valores ajustados --
                Aplicado = ModeloDireto(lbAJ,X0)
                Ajustado = Ajustado.col_join(Aplicado)
            BPTT = sym.diag(BPTT,BPT) #Compondo a matriz bloco para os planos
            CPTT = sym.diag(CPTT,CPT)
            FA = XI - X0
            FAP = FAP.col_join(sym.Matrix([FA[:4]]).T)
        BPTTT = BPTTT.col_join(BPTT) #Compondo a matriz bloco para os planos
        CPTTT = CPTTT.col_join(CPTT)
        BETT = sym.diag(BETT,BET)
        CETT = sym.diag(CETT,CET)
        FA = XI - X0
        fx = fx.col_join(sym.Matrix([FA[4:-4]]).T)
    # -- Concatenar as matrizes derivadas e das funções aplicadas -- 
    BF = BPTTT.row_join(BETT)
    CF = CPTTT.row_join(CETT)
    BF = BF.row_join(BADT)
    CF = CF.row_join(CADT)
    fx = fx.row_insert(0,FAP)
    fx = fx.col_join(sym.Matrix([FA[-4:]]).T)
    fx = np.array(fx).astype(np.float64)
    # -------------------------- Processo de ajustamento --------------------------
    # -- Prepara as metrizes para trabalhar como modelo númerico
    f = -np.array(F -AT*V).astype(np.float64)
    g = -np.array(G -ACT*VC).astype(np.float64)
    AT = np.array(AT).astype(np.float64)
    Q = np.array(Q).astype(np.float64)
    QC = np.array(QC).astype(np.float64)
    WXX = np.array(WXX).astype(np.float64)
    BF = np.array(BF).astype(np.float64)
    CF = np.array(CF).astype(np.float64)
    ACT = np.array(ACT).astype(np.float64)
    WE= inv(AT.dot(Q.dot(np.transpose(AT))))
    WEC= inv(ACT.dot(QC.dot(np.transpose(ACT))))
    # -- CONTRIBUIÇÃO DOS PARAMETROS --
    N = np.transpose(BF).dot(WE.dot(BF))
    T = np.transpose(BF).dot(WE.dot(f))
    # -- CONTRIBUIÇÃO DAS RESTRIÇÕES --
    NC = np.transpose(CF).dot(WEC.dot(CF))
    TC = np.transpose(CF).dot(WEC.dot(g))
    DELTA = inv(N+NC+WXX).dot(T+TC-WXX.dot(fx))
    V = Q.dot(np.transpose(AT).dot(WE.dot(f-BF.dot(DELTA))))
    VC = QC.dot(np.transpose(ACT).dot(WEC.dot(g-CF.dot(DELTA))))
    VX = fx + DELTA
    # -------------------------- Aplicar as correções nos veres iniciasi --------------------------
    CCP = pd.DataFrame() # Vetor para manipuras as corrções do paramentro dos planos
    for g in range (0,vetor.shape[0]):
        cp = pd.DataFrame({'Nx': DELTA[g*4,0],'Ny':DELTA[(g*4)+1,0],'Nz':DELTA[(g*4)+2,0],'d':DELTA[(g*4)+3,0]},index = [plano[g]])
        CCP = CCP.append(cp) 
    vcorrigido = vcorrigido.add(CCP) 
    CCE = pd.DataFrame() # Vetor para manipuras as corrções do paramentro das estações
    for j in range (0,parametros.shape[0]):
        t = ((vetor.shape[0])*4) #Pular os primeiros elementos no vetor das correções que corresponden ao plano
        ce = pd.DataFrame({'omega': DELTA[(t+j*6),0],'kappa':DELTA[(t+j*6)+1,0],'phi':DELTA[(t+j*6)+2,0],'X':DELTA[(t+j*6)+3,0],'Y':DELTA[(t+j*6)+4,0],'Z':DELTA[(t+j*6)+5,0],'a0':DELTA[-4],'b0':DELTA[-3],'b1':DELTA[-2],'c0':DELTA[-1]},index = [estacao[j]])
        CCE = CCE.append(ce)
    parcorrugido = parcorrugido.add(CCE)
    CCL = pd.DataFrame() # Vetor para manipuras as corrções das observacoes
    for k in range (0,observacoes.shape[0]):
        cl = pd.DataFrame({'Estacao':observacoes.index[k],'Plano':[0],'x': V[(k*3),0],'y':V[(k*3)+1,0],'z':V[(k*3)+2,0]})
        CCL = CCL.append(cl)
    CCL.set_index('Estacao', inplace=True)
    obscorrigido = observacoes.add(CCL)
    # -------------------------- Plotar pontos ajustados --------------------------
    # Inicializa a figura para plotagem
    fig = plt.figure() 
    qq = fig.add_subplot(111, projection='3d') 
    # Percorre o vetor para separar em X,Y,Z
    n = 2 # Numero de estações dentro do vetor ajustado
    limite =int(len(Ajustado)/(n*3)) # Marcar o inicio/fim de cada estação
    # Imprimir
    qq.set_xlabel('Eixo X ')
    qq.set_ylabel('Eixo Y ')
    qq.set_zlabel('Eixo Z ')
    for j in range (0,limite):
        xp = float(Ajustado[j*3])
        yp = float(Ajustado[j*3+1])
        zp = float(Ajustado[j*3+2])
        qq.scatter(xp, yp, zp, marker="o", color="blue")
    for k in range (limite,int(len(Ajustado)/3)):
        xp = float(Ajustado[k*3])
        yp = float(Ajustado[k*3+1])
        zp = float(Ajustado[k*3+2])
        qq.scatter(xp, yp, zp, marker="^", color="red")
    plt.savefig(DIR2+teste+"Pontos_"+str(contador)+'.png')
    plt.show() # imprime a imagem
    # -------------------------- Convergencia do sistema --------------------------
    #Teste  do vetor das correções
    if iteracao!= 0:
        for i in range (0, len(DELTA)):
            taxaD = (DELTA[i,0] - DELTA1[i,0])/DELTA[i,0]
            if abs(taxaD) < Vcorrecao:
                MENSAGEIM1 = "Criterio de convergencia"
                MENSAGEM2 ="Teste convergiu para o teste do vetor das correções"
                break
    DELTA1 = DELTA # Armazena a variavel para o teste
    #Teste para o vetor atualizado
    maximo = CCE.max() # busca o maior valor nos fatores de correção
    Mang = max([abs(number) for number in [maximo.omega,maximo.kappa,maximo.phi]])
    Mcoord = max([abs(number) for number in [maximo.X,maximo.Y,maximo.Z]])
    Maps = max([abs(number) for number in [maximo.a0,maximo.b0,maximo.b1,maximo.c0]])
    if (Mang < sc.rad2deg(Lang) and Mcoord < Lcoord and Maps < Laps):
        MENSAGEIM1 = "Criterio de convergencia"
        MENSAGEM2 ="Atingiu as correções minimas"
        break
    if iteracao == limite :
        MENSAGEIM1 = "Criterio de convergencia"
        MENSAGEM2 ="Atingiu o limite de iterações"
    contador = contador+1

# -------------------------- Posteriores --------------------------
# Matriz estatisticas das equações
QXX = inv(N+NC+WXX) # Matriz cofator dos Parametros
WXX = inv(QXX) # Matriz peso dos Parametros
QC = ACT.dot(QC.dot(np.transpose(ACT))) # Matriz cofator das Restrições
WC = inv(QC) # Matriz peso das Restrições
W = np.array(W).astype(np.float64)
var = np.transpose(V).dot(W.dot(V))+np.transpose(VC).dot(WC.dot(VC))+np.transpose(VX).dot(WXX.dot(VX)) # Variancia a posteriori
Qp = AT.dot(Q.dot(np.transpose(AT))) # Matriz cofator das observacoes
Wp = inv(Qp) # Matriz peso das observacoes
# Concatenar as matrizes
WT = np.block([[Wp,       np.zeros((Wp.shape[0],WC.shape[1])),  np.zeros((Wp.shape[0],WXX.shape[1]))],
               [np.zeros((WC.shape[0],Wp.shape[1])),WC, np.zeros((WC.shape[0],WXX.shape[1]))],
               [np.zeros((WXX.shape[0],Wp.shape[1])), np.zeros((WXX.shape[0],WC.shape[1])),WXX ]]) # Matriz TODOS os pesos
QT = np.block([[Q,       np.zeros((Q.shape[0],QC.shape[1])),  np.zeros((Q.shape[0],QXX.shape[1]))],
               [np.zeros((QC.shape[0],Q.shape[1])),QC, np.zeros((QC.shape[0],QXX.shape[1]))],
               [np.zeros((QXX.shape[0],Q.shape[1])), np.zeros((QXX.shape[0],QC.shape[1])),QXX ]])# Matriz TODOS os cofatores

AA = np.block([[AT,       np.zeros((AT.shape[0],ACT.shape[1])),  np.zeros((AT.shape[0],QXX.shape[1]))],
               [np.zeros((ACT.shape[0],AT.shape[1])),ACT, np.zeros((ACT.shape[0],QXX.shape[1]))],
               [np.zeros((QXX.shape[0],AT.shape[1])), np.zeros((QXX.shape[0],ACT.shape[1])),np.eye(QXX.shape[0])]]) # Matriz A com todas as variaveis 
BB = np.concatenate((BF,CF,-np.eye(QXX.shape[0])))
#Calcular a matriz cofator para os resituos

Qvv = (QT.dot(np.transpose(AA).dot(WT.dot(AA.dot(QT)))))-(QT.dot(np.transpose(AA).dot(WT.dot(BB.dot(QXX.dot(np.transpose(BB).dot(WT.dot(AA.dot(QT)))))))))
Qtt = QT-Qvv # Matrix a posteriori

# -------------------------- Teste de variancia --------------------------
# Calcula os valores para o teste estatistico 
conf = 0.01 #intervalo de confiça para o teste estistico 99%
Lsup = st.chi2.ppf(1-conf/2,r)
Linf = st.chi2.ppf(conf/2,r)
# Teste estatistico
quiquadrado = r*var/varp[0]


# -------------------------- Teste de correlação --------------------------
# Matriz dos Coeficinetes de Correlação
# -------------------------- Matrix de coorelação --------------------------
# -- Matriz dos Coeficinetes de Correlação--
# Tem que incrlemntar conforme o numero de planos e estações
corrDF = pd.DataFrame(QXX,columns=['a', 'b', 'c', 'd','a', 'b', 'c', 'd','a', 'b', 'c', 'd','a', 'b', 'c', 'd','a', 'b', 'c', 'd','a', 'b', 'c', 'd','x','y','z','k','λ','ω','x','y','z','k','λ','ω','a0','b0','b1','c0']  )
#columns=['a', 'b', 'c', 'd','a', 'b', 'c', 'd','a', 'b', 'c', 'd','a', 'b', 'c', 'd','a', 'b', 'c', 'd','a', 'b', 'c', 'd','x','y','z','k','λ','ω','x','y','z','k','λ','ω','a0','b0','b1','c0'] 
corrMatrix = corrDF.corr()
# -- Imprimir a matrix --
mask = np.triu(np.ones_like(corrMatrix, dtype=np.bool)) # Opcional
f, ax = plt.subplots(figsize=(14, 12))

#sn.heatmap(corrMatrix.abs(),cmap = "YlGnBu",mask=mask) com a mascara
sn.heatmap(corrMatrix.abs(),cmap = "YlGnBu")
plt.savefig(DIR2+teste+"COOR.png")
plt.show()

print(MENSAGEM2)
# -------------------------- Teste de quiquadrado --------------------------

if (Linf < quiquadrado[0] and quiquadrado[0] < Lsup):
    print("A hipótese basica, sigma a priori ser igual ao posteriori, não é rejeitada ")
else:
    print("A hipótese basica, sigma a priori ser igual ao posteriori,é rejeitada ")

# -------------------------- Salva arquivo com as correções finais --------------------------
obscorrigido.to_csv(DIR2+teste+"OBS_CORRIGIDA.csv", sep=";")
parcorrugido.to_csv(DIR2+teste+"PAR_CORRIGIDA.csv", sep=";")
vcorrigido.to_csv(DIR2+teste+"VET_CORRIGIDA.csv", sep=";")
corrMatrix.to_csv(DIR2+teste+'CORR_Matrix.csv',sep=";")
np.savetxt(DIR2+teste+"QTT.csv", Qtt, delimiter=";")
np.savetxt(DIR2+teste+"VARIANCIA.csv", varp, delimiter=";")
np.savetxt(DIR2+teste+"QC.csv", QC, delimiter=";")
np.savetxt(DIR2+teste+"DELTA.csv", DELTA, delimiter=";")
np.savetxt(DIR2+teste+"V.csv", V, delimiter=";")
np.savetxt(DIR2+teste+"VX.csv", VX, delimiter=";")
np.savetxt(DIR2+teste+"VC.csv", VC, delimiter=";")


