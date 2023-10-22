import numpy as np
import matplotlib.pyplot as plt

# Função para calcular a densidade de probabilidade da distribuição alvo (normal com média mu e desvio padrão sigma)
def target(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

# Função para calcular a log-verossimilhança das amostras observadas
# Função para calcular a log-verossimilhança das amostras observadas
def log_likelihood(z_obs, H_obs, error_obs, mu, sigma):
    n = len(z_obs)
    log_like = np.zeros(n)
    for i in range(n):
        log_like[i] = -0.5 * np.log(2 * np.pi * error_obs[i]**2)
        log_like[i] -= 0.5 * (((H_obs[i] - target(z_obs[i], mu, sigma)) / error_obs[i])**2)
    return np.sum(log_like)


# Função de transição proposta (normal com média x e desvio padrão 1)
def proposal(x):
    return np.random.normal(x, 1)

# Ler as amostras do arquivo .txt
data = np.loadtxt('inputdata.txt')
z_obs = data[:, 0]  # Valores observados de z
H_obs = data[:, 1]  # Valores observados de H(z)
error_obs = data[:, 2]  # Erros correspondentes

# Definir o número de amostras a serem geradas
n_samples = 10000

# Definir a amostra inicial e a sequência de amostras geradas
mu_initial = np.random.normal(0, 1)  # Amostra inicial para o parâmetro mu
sigma_initial = np.random.uniform(0, 2)  # Amostra inicial para o parâmetro sigma
log_like_initial = log_likelihood(z_obs, H_obs, error_obs, mu_initial, sigma_initial)

mu = np.zeros(n_samples)
sigma = np.zeros(n_samples)
log_like = np.zeros(n_samples)
mu[0] = mu_initial
sigma[0] = sigma_initial
log_like[0] = log_like_initial

# Gerar a sequência de amostras usando o algoritmo de Metropolis-Hastings
for i in range(1, n_samples):
    mu_proposed = proposal(mu[i-1])
    sigma_proposed = proposal(sigma[i-1])
    log_like_proposed = log_likelihood(z_obs, H_obs, error_obs, mu_proposed, sigma_proposed)
    
    log_acceptance = log_like_proposed - log_like[i-1]
    log_acceptance += np.log(target(mu[i-1], 0, 1)) - np.log(target(mu_proposed, 0, 1))
    log_acceptance += np.log(target(sigma[i-1], 0, 2)) - np.log(target(sigma_proposed, 0, 2))
    
    accept = np.log(np.random.uniform(size=1)) < log_acceptance
    if accept:
        mu[i] = mu_proposed
        sigma[i] = sigma_proposed
        log_like[i] = log_like_proposed
    else:
        mu[i] = mu[i-1]
        sigma[i] = sigma[i-1]
        log_like[i] = log_like[i-1]

# Plotar as amostras da cadeia de Markov gerada para o parâmetro mu
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(mu)
plt.xlabel('Iteração')
plt.ylabel('mu')
plt.title('Cadeia de Markov para mu')

# Plotar as amostras da cadeia de Markov gerada para o parâmetro sigma
plt.subplot(1, 2, 2)
plt.plot(sigma)
plt.xlabel('Iteração')
plt.ylabel('sigma')
plt.title('Cadeia de Markov para sigma')

plt.tight_layout()
plt.show()
