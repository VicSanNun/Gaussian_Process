from getdist import plots, MCSamples
import getdist
import numpy as np
import emcee
import matplotlib.pyplot as plt
import time
import corner

# matter_density = 0.3
# lambda_density = 0.7
# H_0 = 70

def make_vector(dim, vec_inicial):
    vec=np.zeros([1,len(dim)],dtype=np.float32)
    for a in range(len(dim)):
        vec[0,a]=vec_inicial[a]
    return vec

#Função de log-verossimilhança para o ajuste
def ln_likelihood(params, z, Dvec, error):
    Chi2T=0.
    sigma, l, H_0, matter_density = params

    #lambda_density = np.subtract(1.0, matter_density)

    # matter_density = 0.3
    # lambda_density = 0.7
    # H_0 = 70

    Dvecdif=Dvec*0.0
    Dvecdif =Dvec-H_0*np.sqrt(matter_density*(1+z)**3+(1-matter_density))
    
    #Dvecdif =Dvec

    mbvectot=Dvecdif
    mbvectotT=mbvectot.T

    covariance_matrix = sigma**2 * np.exp(-0.5 * (z[:, None] - z[None, :])**2 / l**2)
    covariance_matrix += np.diag(error**2)
    Icov = np.linalg.inv(covariance_matrix)

    Chi2T=(mbvectot.dot(Icov)).dot(mbvectotT)

    try:
        cho_factor = np.linalg.cholesky(covariance_matrix)
    except np.linalg.LinAlgError:
        return -np.inf
    
    #chi2 = np.sum(np.linalg.solve(cho_factor.T, Hz).T * Hz)
    return -0.5 * Chi2T - np.sum(np.log(np.diagonal(cho_factor)))-float(len(z))/2.*(np.log(2.*3.1415926))

# Priori3
#aumentar range sigma
def ln_prior(params):
    sigma, l, H_0, matter_density = params
    if 0.0 < sigma < 10 and 0.0 < l < 10 and 0.0 < H_0 < 100 and 0.0 < matter_density < 1:
        return 0.0
    return -np.inf

#Posteriori (log-probabilidade)
def ln_posterior(params, z, Dvec, error):
    lp = ln_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(params, z, Dvec, error)

#Carregando dados do arquivo txt
data = np.loadtxt("inputdata.txt")
z, Hz, error = data[:, 0], data[:, 1], data[:, 2]
Dvec = make_vector(z, Hz)

print('Dados Carregados')

#Chute inicial para sigma e l
initial_params = [1.0, 1.0, 1.0, 1.0]

#Número de caminhantes de MCMC
n_walkers = 50

#Número de passos de MCMC
n_steps = 1000

#Inicializando caminhantes
pos = initial_params + 1e-4 * np.random.randn(n_walkers, len(initial_params))

#Criando o sampler de MCMC
sampler = emcee.EnsembleSampler(n_walkers, len(initial_params), ln_posterior, args=(z, Dvec, error))
print('Sampler criado')

print('Executando o MCMC')
start_time = time.time()

# Executando o MCMC
sampler.run_mcmc(pos, n_steps)

end_time = time.time()

elapsed_time = end_time - start_time
print("O MCMC rodou em: {:.2f} segundos".format(elapsed_time))

# Obtendo as cadeias de amostras
samples = sampler.get_chain(discard=100, thin = 15, flat=True)
print('Cadeias de amostras obtidas')

np.savetxt("amostras_sigma.txt", samples[:, 0])
np.savetxt("amostras_l.txt", samples[:, 1])
np.savetxt("amostras_l_sigma_H0_omega_m.txt", samples)

print('Cadeias de amostras gravadas em arquivo')

# Obtendo os valores estimados para sigma e l
sigma_est, l_est, H0_est, omega_m_est = np.median(samples, axis=0)
print('Encontrada as medianas do Sigma e do L')

print("Sigma estimado:", sigma_est)
print("l estimado:", l_est)
print("H0 estimado:", H0_est)
print("\Omega_m estimado:", omega_m_est)
print("\Omega_\Lambda estimado:", (1-omega_m_est))

# Plotando o histograma dos valores de sigma
plt.hist(samples[:, 0], bins=30, density=True, alpha=0.5)
plt.axvline(sigma_est, color='red', linestyle='dashed', linewidth=2, label=f'Valor estimado = {sigma_est}')
plt.xlabel('Sigma')
plt.ylabel('Densidade')
plt.legend()
plt.savefig("histograma_sigma.png")  # Salvando o histograma em um arquivo PNG
plt.show()

# Plotando o histograma dos valores de l
plt.hist(samples[:, 1], bins=30, density=True, alpha=0.5)
plt.axvline(l_est, color='red', linestyle='dashed', linewidth=2, label=f'Valor estimado = {l_est}')
plt.xlabel('l')
plt.ylabel('Densidade')
plt.legend()
plt.savefig("histograma_l.png")  # Salvando o histograma em um arquivo PNG
plt.show()

# Plotando o histograma dos valores de l
plt.hist(samples[:, 2], bins=30, density=True, alpha=0.5)
plt.axvline(H0_est, color='red', linestyle='dashed', linewidth=2, label=f'Valor estimado = {H0_est}')
plt.xlabel('H_0')
plt.ylabel('Densidade')
plt.legend()
plt.savefig("histograma_H0.png")  # Salvando o histograma em um arquivo PNG
plt.show()


# Plotando o histograma dos valores de l
plt.hist(samples[:, 3], bins=30, density=True, alpha=0.5)
plt.axvline(omega_m_est, color='red', linestyle='dashed', linewidth=2, label=f'Valor estimado = {omega_m_est}')
plt.xlabel('\Omega_m')
plt.ylabel('Densidade')
plt.legend()
plt.savefig("histograma_Omega_m.png")  # Salvando o histograma em um arquivo PNG
plt.show()

names1 = ['\sigma','l', 'H_0', '\Omega_m']
labels1 =  names1#['\Omega_m','w_b','b','H_0']
nomearq="mtrN"

mult_order=2

samples = MCSamples(samples = samples ,names = names1, labels = labels1, settings={'boundary_correction_order':1, 
	                          'mult_bias_correction_order':1,'ignore_rows':0.4})

name='samples2_PAN_LCDM'
g = plots.get_subplot_plotter()
g.triangle_plot([samples],names1,filled=True)
g.export('triangle'+name+'.pdf')