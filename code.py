import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

np.random.seed(1234)

def generate_beta_sample(alpha, beta, n=100):
    sample = np.random.beta(alpha, beta, n)
    return sample

def make_quantile_bins(x, N):
    q = np.linspace(0, 100, N + 1)
    return np.percentile(x, q)

def generate_response(sample, sigma=1):
    m_x = np.sin((sample / 3 + 0.1) ** (-1))
    noise = np.random.normal(0, sigma, len(sample))
    response = m_x + noise
    return np.vstack((sample, response))

def fit_polynomial(sample, N, bins, p=4):
    x, y  = sample
    bin_id = np.digitize(x, bins, right=True) - 1
    coeffs = []
    for j in range(N):
        mask = bin_id == j
        if mask.sum() > 4:
            coeffs.append(np.polyfit(x[mask], y[mask], p))
        else:
            coeffs.append(None)
    return coeffs

def compute_theta(sample, betas, bins, N):
    x = sample[0]
    n = x.size

    bin_id = np.digitize(x, bins, right=True) - 1

    m2 = np.zeros(n)
    for j in range(N):
        idx = bin_id == j
        coeffs = betas[j]
        if coeffs is None:
            continue
        b4, b3, b2 = coeffs[:3]
        xi = x[idx]
        m2[idx] = 12*b4*xi**2 + 6*b3*xi + 2*b2

    return np.mean(m2**2)

def compute_sigma(sample, betas, bins):
    x, y = sample
    n = y.size
    N = len(betas)

    bin_id = np.digitize(x, bins, right=True) - 1

    resid2_sum = 0.0
    for j in range(N):
        coeffs = betas[j]
        mask = bin_id == j
        a, b, c, d, e = coeffs
        y_hat = ((a*x[mask] + b)*x[mask] + c)*x[mask]**2 + d*x[mask] + e
        resid2_sum += np.sum((y[mask] - y_hat)**2)

    df   = n - 5*N
    return np.sqrt(resid2_sum / df)

def compute_h_amise(n, sigma, x, theta):
    support_length = np.max(x) - np.min(x)
    h_amise = n**(-1/5) * ((35 * sigma**2 * support_length) / theta)**(1/5)
    return h_amise

def solve_h(alpha, beta, n, N, sigma):
    sample_x = generate_beta_sample(alpha, beta, n)
    bins = make_quantile_bins(sample_x, N)
    sample = generate_response(sample_x, sigma)
    betas = fit_polynomial(sample, N, bins)
    theta = compute_theta(sample, betas, bins, N)
    sigma = compute_sigma(sample, betas, bins)
    h_amise = compute_h_amise(n, sigma, sample_x, theta)
    return h_amise



# Mallow CP

def rss_df_given_dataset(x, y, N):
    n = len(y)
    if N < 1:
        raise ValueError("N must be >= 1")
    edges = np.percentile(x, np.linspace(0, 100, N + 1))
    internal = edges[1:-1]
    bin_id = np.digitize(x, internal, right=False)

    rss = 0.0
    for j in range(N):
        mask = (bin_id == j)
        coeffs = np.polyfit(x[mask], y[mask], deg=4)
        y_hat = np.polyval(coeffs, x[mask])
        rss += np.sum((y[mask] - y_hat) ** 2)

    df = n - 5 * N
    if df <= 0:
        raise ValueError(f"Nonpositive degrees of freedom (df={df}). Reduce N or increase n.")
    return rss, df


def mallow_cp_for_dataset(x, y, N_grid):
    n = len(y)
    N_max = max(min(n // 20, 5), 1)
    rss_Nmax, df_Nmax = rss_df_given_dataset(x, y, N_max)
    sigma2_hat = rss_Nmax / df_Nmax

    cp_vals = []
    for N in N_grid:
        rss_N, _ = rss_df_given_dataset(x, y, N)
        cp_N = rss_N / sigma2_hat - (n - 10 * N)
        cp_vals.append(cp_N)
    return np.array(cp_vals), N_max

def plot_scatter_plots():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sample_x = generate_beta_sample(alpha, beta, n)
    sample_no_noise = generate_response(sample_x, 0)
    sample = generate_response(sample_x, sigma)

    # Plot without noise
    axes[0].scatter(sample_no_noise[0], sample_no_noise[1], alpha=0.5)
    axes[0].set_title('No Noise')
    axes[0].set_xlabel('Sample from Beta Distribution')
    axes[0].set_ylabel('$m(x)$')

    # Plot with noise
    axes[1].scatter(sample[0], sample[1], alpha=0.5)
    axes[1].set_title('With Noise')
    axes[1].set_xlabel('Sample from Beta Distribution')
    axes[1].set_ylabel('$m(x) + \epsilon$')

    plt.tight_layout()
    plt.savefig("scatter_plots.png", dpi=600)

def optimal_N_for_n(alpha, beta_param, n, sigma, N_grid = np.arange(1, 6), reps = 5):
    cp_accum = []
    for r in range(reps):
        x = generate_beta_sample(alpha, beta_param, n)
        x, y = generate_response(x, sigma)
        cp_vals, N_max = mallow_cp_for_dataset(x, y, N_grid)
        cp_accum.append(cp_vals)
    
    mean_cp = np.mean(np.vstack(cp_accum), axis=0)
    N_staar = int(N_grid[np.argmin(mean_cp)])
    return N_staar, mean_cp


alpha = 2
beta = 5
n = 1000
N = 10
sigma = 0.5

plot_scatter_plots()



n_values = [1000, 10000, 100000]

safe_N_max = 15
N_grid = np.arange(1, safe_N_max + 1)

# Plot of optimal bandwidth vs N for different n
plt.figure(figsize=(8, 5))
for n in n_values:
    h_vals = []
    for N in N_grid:
        try:
            h_vals.append(float(solve_h(alpha, beta, n, N, sigma)))
        except ValueError:
            h_vals.append(np.nan)
    plt.plot(N_grid, h_vals, label=f"n = {n}")

plt.xlabel("N (number of blocks/bins)")
plt.yscale("log")
plt.ylabel("$h_{AMISE}$ (log scale)")
plt.title("Optimal bandwidth vs N for different n")
plt.legend(title="Sample size")
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig("optimal_bandwidth.png", dpi=600)


# plot of Mallow's Cp vs N for different n
plt.figure(figsize=(8, 5))
for n in n_values:
    x_n = generate_beta_sample(alpha, beta, n)
    x_n, y_n = generate_response(x_n, sigma)
    cp_vals, N_max = mallow_cp_for_dataset(x_n, y_n, N_grid)
    plt.plot(N_grid, cp_vals, label=f"n = {n}")

plt.xlabel('N (number of bins)')
plt.yscale('log')
plt.ylabel("Mallow's $C_p$ (log scale)")
plt.title("Mallow's $C_p$ vs N for different n")
plt.legend(title='Sample size')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mallows_cp.png", dpi=600)




N_list       = [1, 2, 5]
grid_npts    = 1000
scatter_pts  = 5000

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, n_sample in zip(axes, [1000, 10000000]):   # two sample sizes
    x_sample = generate_beta_sample(alpha, beta, n_sample)
    x, y     = generate_response(x_sample, sigma)

    x_grid = np.linspace(0, 1, grid_npts)
    m_true = np.sin((x_grid/3 + 0.1)**(-1))

    n_show = min(scatter_pts, n_sample)
    idx = np.random.choice(n_sample, size=n_show, replace=False)
    ax.scatter(x[idx], y[idx], s=6, alpha=0.15, color="grey")

    ax.plot(x_grid, m_true, lw=2.5, color="black", label="true $m(x)$")

    colors = ["tab:blue", "tab:orange", "tab:red"]
    for N, col in zip(N_list, colors):
        bins  = make_quantile_bins(x_sample, N)
        betas = fit_polynomial((x, y), N, bins, p=4)

        y_hat  = np.full_like(x_grid, np.nan, dtype=float)
        bin_id = np.digitize(x_grid, bins, right=True) - 1
        for j in range(N):
            coeffs = betas[j]
            if coeffs is None:
                continue
            mask = bin_id == j
            y_hat[mask] = np.polyval(coeffs, x_grid[mask])

        ax.plot(x_grid, y_hat, lw=2, color=col,
                label=f"fitted ($N={N}$)", alpha=0.7)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$\hat m(x)$")
    ax.set_title(f"n = {n_sample:,},  σ = {sigma}")
    ax.grid(alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True)

fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("density.png", dpi=600)





n_fixed = 10_000
N_fixed = 2
sigma_fixed = 0.5

alpha_grid = np.linspace(0.5, 10.0, 10)
beta_grid  = np.linspace(0.5, 10.0, 10)

H = np.empty((len(alpha_grid), len(beta_grid)))
for i, a in enumerate(alpha_grid):
    for j, b in enumerate(beta_grid):
        H[i, j] = solve_h(a, b, n=n_fixed, N=N_fixed, sigma=sigma_fixed)

plt.figure(figsize=(7, 6))
im = plt.imshow(
    H, origin='lower', aspect='auto',
    extent=[beta_grid.min(), beta_grid.max(), alpha_grid.min(), alpha_grid.max()]
)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\alpha$')
plt.title(r'$\hat h_{\mathrm{AMISE}}$ heatmap (n=10{,}000, N=2)')
cbar = plt.colorbar(im)
cbar.set_label(r'$\hat h_{\mathrm{AMISE}}$')

plt.tight_layout()
plt.savefig("h_amise_heatmap_n10000_N2.png", dpi=600)


# Analysis of optimal N vs n
n_grid = np.unique(np.logspace(3, 6, num = 20, dtype=int))
N_grid = np.arange(1, 6)
reps = 10

N_star_list = []
for n in n_grid:
    N_star, cp_vals = optimal_N_for_n(alpha, beta, n, sigma, N_grid, reps)
    N_star_list.append(N_star)

plt.figure(figsize=(8, 5))
plt.plot(n_grid, N_star_list, marker='o')
plt.xscale('log')
plt.yticks([1, 2, 3, 4, 5])
plt.xlabel('n (log scale)')
plt.ylabel('Optimal N (argmin Mallows $C_p$)')
plt.title(f'Optimal N vs n (averaged over {reps} replicates)')
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig("optimal_N_vs_n.png", dpi=600)





N_grid = np.arange(2, 6)

sample_x = generate_beta_sample(alpha, beta, n)
sample    = generate_response(sample_x, sigma)

sigma_hats = []
theta_hats = []

for N_ in N_grid:
    bins  = make_quantile_bins(sample_x, N_)
    betas = fit_polynomial(sample, N_, bins, p=4)

    theta_hat = compute_theta(sample, betas, bins, N_)
    sigma_hat = compute_sigma(sample, betas, bins)

    theta_hats.append(theta_hat)
    sigma_hats.append(sigma_hat)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

axes[0].plot(N_grid, sigma_hats, marker='o')
axes[0].set_title(r'$\hat{\sigma}$ vs $N$')
axes[0].set_xlabel('N (number of bins)')
axes[0].set_ylabel(r'$\hat{\sigma}$')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 1)

axes[1].plot(N_grid, theta_hats, marker='s', linestyle='--')
axes[1].set_title(r'$\hat{\theta}_{22}$ vs $N$')
axes[1].set_xlabel('N (number of bins)')
axes[1].set_ylabel(r'$\hat{\theta}_{22} = \mathbb{E}[\hat m^{\prime\prime}(X)^2]$')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 40000)

plt.tight_layout()
plt.savefig("sigma_theta_vs_N_side_by_side.png", dpi=600)
plt.show()
