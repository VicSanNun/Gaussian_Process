"""
This is an example how to use the dgp module of GaPP.
You can run it with 'python dgp_example.py'.
"""


from gapp import dgp
import numpy as np
from numpy import loadtxt, savetxt
import random

if __name__=="__main__":

    train_hyperparameters =  False

    def gerar_amostras_aleatorias(nome_arquivo, quantidade_amostras):
        try:
            # Lendo o conteúdo do arquivo original
            with open(nome_arquivo, "r") as file:
                linhas = file.readlines()

            # Removendo o cabeçalho (se houver)
            cabeçalho = linhas[0] if linhas[0].startswith("l") else ""
            linhas = linhas[1:]

            # Verificando se há pelo menos a quantidade desejada de amostras para sortear
            if len(linhas) < quantidade_amostras:
                print("O arquivo original não contém a quantidade mínima de amostras desejada.")
                return False

            # Sorteando as linhas aleatórias
            amostras_aleatorias = random.sample(linhas, quantidade_amostras)

            # Adicionando o cabeçalho de volta às amostras
            amostras_aleatorias_com_cabeçalho = cabeçalho + "".join(amostras_aleatorias)

            # Escrevendo as amostras aleatórias em um novo arquivo
            nome_arquivo_amostras_aleatorias = f"{nome_arquivo.split('.')[0]}_amostras_aleatorias.txt"
            with open(nome_arquivo_amostras_aleatorias, "w") as output_file:
                output_file.write(amostras_aleatorias_com_cabeçalho)

            print(f"Amostras aleatórias foram salvas em {nome_arquivo_amostras_aleatorias}")
            return True
        
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            return False

    # load the data from inputdata.txt
    #mudei o inputdata. Coloquei os dados que o prof mandou
    (X, Y, Sigma) = loadtxt("./inputdata.txt", unpack='True')

    # nstar points of the function will be reconstructed 
    # between xmin and xmax
    #modifiquei o valor de xmax para 2.5 o valor original era 10
    xmin = 0.0
    xmax = 2.5
    nstar = 200

    #Lambda CDM Parameters
    matter_density = 0.3
    lambda_density = 0.7
    H_0 = 70

    def lambda_CDM(z):
        H = H_0*np.sqrt(matter_density*(1+z)**3+lambda_density)
        return H

    def dlambda_CDM(z):
        dH = (3*matter_density*(z+1)**2)/(2*np.sqrt(matter_density*(1+z)**3+lambda_density))
        return dH

    def zero(z):
        return 0

    # initialization of the Gaussian Process
    g = dgp.DGaussianProcess(X, Y, Sigma, cXstar=(xmin, xmax, nstar), mu=zero, dmu=zero)
    
    if (train_hyperparameters):
        #g = dgp.DGaussianProcess(X, Y, Sigma, cXstar=(xmin, xmax, nstar))
        # initial values of the hyperparameters
        #o theta inicial vai ser os valores do output do MCMC
        initheta = [2.0, 2.0]

        # training of the hyperparameters and reconstruction of the function
        (rec, theta) = g.gp(theta=initheta)

        # reconstruction of the first, second and third derivatives.
        # theta is fixed to the previously determined value.
        (drec, theta) = g.dgp(thetatrain='False')

        #save the output
        savetxt("f.txt", rec)
        savetxt("df.txt", drec)

        #no arquivo plot.py modifiquei os valores de xmax para 2.5 o original era 10
        import plot
        # plot.plot(X, Y, Sigma, rec, drec, "plot2.png")
        plot.plot(X, Y, Sigma, rec, drec, "plot1.png")
    else:
        nmr_amostras = 100
        gerar_amostras_aleatorias('amostras_l_sigma_H0_omega_m.txt', nmr_amostras)
        (l, sigma, H0, wm) = loadtxt("./amostras_l_sigma_H0_omega_m_amostras_aleatorias.txt", unpack='True')
        
        for i in range(nmr_amostras):
            l_amostra = l[i]
            sigma_amostra = sigma[i]
            H0_amostra = H0[i]
            wm_amostra = wm[i]
            
            def lambda_CDM(z):
                H = H0_amostra*np.sqrt(wm_amostra*(1+z)**3+(1-wm_amostra))
                return H

            def dlambda_CDM(z):
                dH = (3*wm_amostra*(z+1)**2)/(2*np.sqrt(wm_amostra*(1+z)**3+(1-wm_amostra)))
                return dH

            g = dgp.DGaussianProcess(X, Y, Sigma, cXstar=(xmin, xmax, nstar), mu=lambda_CDM, dmu=dlambda_CDM)
            
            initheta = [l_amostra, sigma_amostra]

            (rec, theta) = g.gp(theta=initheta, thetatrain='False')
            (drec, theta) = g.dgp(thetatrain='False')

            savetxt(f"rec/f_{i}.txt", rec)
            savetxt(f"drec/df_{i}.txt", drec)



