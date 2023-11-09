import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt, savetxt
import os

def generate_gaussian(Hz, erroZ, num_points=1000):
    mean = Hz
    std_dev = erroZ
    gaussian_samples = np.random.normal(mean, std_dev, num_points)
    return gaussian_samples

def save(dist, i, j):
    if not os.path.exists(f"./gaussian/f_{i}/"):
        # Se o diretório não existir, cria-o
        os.makedirs(f"./gaussian/f_{i}/")

    savetxt(f"gaussian/f_{i}/z_{j}.txt", dist)

def concatenate():
    #file = []
    for i in range(200):
        file = []
        for j in range(100):

            print(f'Abrindo aruivo ./gaussian/f_{j}/z_{i}.txt"')

            with open(f"./gaussian/f_{j}/z_{i}.txt", 'r') as arquivo_entrada:
                conteudo = arquivo_entrada.read()

            file.append(conteudo)

        print(f'Salvando gaussianas_concatenadas/z_{i}.txt')

        if not os.path.exists(f"./gaussianas_concatenadas/"):
        # Se o diretório não existir, cria-o
            os.makedirs(f"./gaussianas_concatenadas/")

        np.savetxt(f"./gaussianas_concatenadas/z_{i}.txt", [file], fmt='%s')

#Gerando as Gaussianas
for i in range(100):
    (z, Hz, Sigma) = loadtxt(f"./rec/f_{i}.txt", unpack='True')
    for j in range(len(z)):
        gaussian_samples = generate_gaussian(Hz[j], Sigma[j], num_points=1000)
        save(gaussian_samples, i, j)

#concatenando as gaussianas
concatenate()

#Extraindo média e desvio as Gaussianas
dado = []
for i in range(200):
    with open(f"./gaussianas_concatenadas/z_{i}.txt", "r") as arquivo:
        dados = []

        for linha in arquivo:
            try:
                dados.append(float(linha.strip()))
            except:
                print('Erro')
            
        dados_array = np.array(dados)

        media = np.mean(dados_array)
        desvio_padrao = np.std(dados_array)

    dado.append(f'{i} {media} {desvio_padrao} \n')
    print(f'Para Z_{i} temos média {media} e sigma {desvio_padrao}')
np.savetxt(f"./outupt_{i}.txt", [dado], fmt='%s')


with open('outupt_199.txt', 'r') as file:
    lines = file.readlines()

# Processando os valores da coluna 'z' e atualizando para o novo intervalo
new_lines = []
for line in lines:
    data = line.split()
    if data:  # Verifica se há dados na linha
        z_value = int(data[0])
        new_z = z_value * (2.5 / 199)  # Convertendo o valor de z para o novo intervalo
        data[0] = str(new_z)
        new_line = '\t'.join(data)
        new_lines.append(new_line + '\n')

# Salvando os dados modificados em um novo arquivo
with open('dados_modificados.txt', 'w') as new_file:
    new_file.writelines(new_lines)


(z, Hz, Sigma) = loadtxt("./dados_modificados.txt", unpack='True')
(z_init, Hz_init, Sigma_init) = loadtxt("./inputdata.txt", unpack='True')

plt.plot(z,Hz)
plt.fill_between(z, Hz-Sigma, Hz+Sigma, facecolor='grey', alpha=0.5)
plt.errorbar(z_init, Hz_init, yerr=Sigma_init, label='Dados com Erro', lw=0.2, color='red', fmt='ro')
plt.xlabel('Z')
plt.ylabel('Hz')
#plt.legend()
plt.grid(False)

plt.savefig('plot5.png')
# Mostrar o gráfico
plt.show()
        