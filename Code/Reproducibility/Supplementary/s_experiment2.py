import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define parameters
num_samples = 100_000
domain = np.linspace(0, 6, 500)

def generate_unimodal():
    mu = np.random.uniform(0.5, 5.5)
    sigma = np.random.uniform(0.2, 1.0)
    pdf = norm.pdf(domain, loc=mu, scale=sigma)
    dx = domain[1] - domain[0]
    return pdf / np.sum(pdf * dx)

def generate_bimodal():
    mu1 = np.random.uniform(0.5, 2.5)
    mu2 = np.random.uniform(3.5, 5.5)
    sigma1 = np.random.uniform(0.2, 1.0)
    sigma2 = np.random.uniform(0.2, 1.0)
    weight = np.random.uniform(0.3, 0.7)
    pdf = weight * norm.pdf(domain, loc=mu1, scale=sigma1) + (1 - weight) * norm.pdf(domain, loc=mu2, scale=sigma2)
    dx = domain[1] - domain[0]
    return pdf / np.sum(pdf * dx)

# Generate all distributions
unimodal_samples = np.array([generate_unimodal() for _ in range(num_samples)])
bimodal_samples = np.array([generate_bimodal() for _ in range(num_samples)])


def compute_sharpness(pdf):
    d_sorted = np.sort(pdf)
    N = d_sorted.size
    L = domain[-1] - domain[0]
    v = L / N
    weights = np.arange(N) + 0.5
    t = weights * v
    integral = v * np.dot(d_sorted, t)
    return (2.0 / L) * integral - 1.0

# Apply sharpness computation
sharpness_unimodal = np.array([compute_sharpness(p) for p in unimodal_samples])
sharpness_bimodal = np.array([compute_sharpness(p) for p in bimodal_samples])

# Filter for sharpness ≈ 0.6 within delta
delta_sharp = 0.01
mask_uni = np.abs(sharpness_unimodal - 0.6) <= delta_sharp
mask_bi = np.abs(sharpness_bimodal - 0.6) <= delta_sharp

# Extract matching distributions
unimodal_sharp_0_6 = unimodal_samples[mask_uni]
bimodal_sharp_0_6 = bimodal_samples[mask_bi]


def compute_variance(pdf):
    dx = domain[1] - domain[0]
    mean = np.sum(pdf * domain * dx)
    return np.sum(pdf * (domain - mean)**2 * dx)

# Compute variance for each filtered set
variance_uni = np.array([compute_variance(p) for p in unimodal_sharp_0_6])
variance_bi = np.array([compute_variance(p) for p in bimodal_sharp_0_6])

# Find index of min and max variance in each
idx_uni_min, idx_uni_max = np.argmin(variance_uni), np.argmax(variance_uni)
idx_bi_min, idx_bi_max = np.argmin(variance_bi), np.argmax(variance_bi)

# Extract corresponding distributions
distributions_to_plot = {
    "Unimodal (Lowest Var)": unimodal_sharp_0_6[idx_uni_min],
    "Unimodal (Highest Var)": unimodal_sharp_0_6[idx_uni_max],
    "Bimodal (Lowest Var)": bimodal_sharp_0_6[idx_bi_min],
    "Bimodal (Highest Var)": bimodal_sharp_0_6[idx_bi_max],
}

# Plot
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axs = axs.flatten()

for ax, (title, pdf) in zip(axs, distributions_to_plot.items()):
    ax.plot(domain, pdf)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('Density')

plt.tight_layout()
plt.show()


def compute_entropy(pdf):
    dx = domain[1] - domain[0]
    return -np.sum(np.where(pdf > 0, pdf * np.log(pdf), 0) * dx)

entropy_uni = np.array([compute_entropy(p) for p in unimodal_sharp_0_6])
entropy_bi = np.array([compute_entropy(p) for p in bimodal_sharp_0_6])

# Find highest and lowest entropy indices
idx_uni_min_ent, idx_uni_max_ent = np.argmin(entropy_uni), np.argmax(entropy_uni)
idx_bi_min_ent, idx_bi_max_ent = np.argmin(entropy_bi), np.argmax(entropy_bi)

# Extract corresponding distributions
entropy_distributions_to_plot = {
    "Unimodal (Lowest Entropy)": unimodal_sharp_0_6[idx_uni_min_ent],
    "Unimodal (Highest Entropy)": unimodal_sharp_0_6[idx_uni_max_ent],
    "Bimodal (Lowest Entropy)": bimodal_sharp_0_6[idx_bi_min_ent],
    "Bimodal (Highest Entropy)": bimodal_sharp_0_6[idx_bi_max_ent],
}

# Plot
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axs = axs.flatten()

for ax, (title, pdf) in zip(axs, entropy_distributions_to_plot.items()):
    ax.plot(domain, pdf)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('Density')

plt.tight_layout()
plt.show()


target_sharpness = 0.4
mask_bi_04 = np.abs(sharpness_bimodal - target_sharpness) <= delta_sharp
bimodal_sharp_0_4 = bimodal_samples[mask_bi_04]

# Compute entropy for sharpness ≈ 0.4 sets
entropy_bi_04 = np.array([compute_entropy(p) for p in bimodal_sharp_0_4])

# Identify min and max entropy indices
idx_bi_min_ent_04, idx_bi_max_ent_04 = np.argmin(entropy_bi_04), np.argmax(entropy_bi_04)

# Extract distributions
entropy_distributions_04 = {
    "Bimodal (Lowest Entropy)": bimodal_sharp_0_4[idx_bi_min_ent_04],
    "Bimodal (Highest Entropy)": bimodal_sharp_0_4[idx_bi_max_ent_04],
}

# Plot
fig, axs = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True)
axs = axs.flatten()

for ax, (title, pdf) in zip(axs, entropy_distributions_04.items()):
    ax.plot(domain, pdf)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('Density')

plt.tight_layout()
plt.show()



target_entropy_6 = 1
delta_sharp = 0.05

entropy_all_bimodal = np.array([compute_entropy(p) for p in bimodal_samples])

mask_bi_ent_6 = np.abs(entropy_all_bimodal - target_entropy_6) <= delta_sharp

bimodal_ent_6 = bimodal_samples[mask_bi_ent_6]

sharpness_bi_ent_6 = np.array([compute_sharpness(p) for p in bimodal_ent_6])

# Indices of min and max sharpness
idx_bi_min_s_6, idx_bi_max_s_6 = np.argmin(sharpness_bi_ent_6), np.argmax(sharpness_bi_ent_6)

# Distributions to plot
sharpness_distributions_ent_6 = {
    "Bimodal (Lowest Sharpness)": bimodal_ent_6[idx_bi_min_s_6],
    "Bimodal (Highest Sharpness)": bimodal_ent_6[idx_bi_max_s_6],
}

# Plot
fig, axs = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True)
axs = axs.flatten()

for ax, (title, pdf) in zip(axs, sharpness_distributions_ent_6.items()):
    ax.plot(domain, pdf)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('Density')

plt.tight_layout()
plt.show()