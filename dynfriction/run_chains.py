import numpy as np
import h5py as h5
from os.path import join
from dynfriction import DF
from dynhalo.corrfunc.model import power_law
from mcmc import run_chain


def DF_lkl(pars, *data):
    alpha, beta, lgsigma_int = pars
    ratio_data, cov, n_threads = data
    if alpha < -8 or alpha > -5 or beta < -6 or beta > -4 or lgsigma_int < -4 or lgsigma_int > 0:
        return -np.inf
    return DF_mdpl2.get_lglkl(alpha, beta, 10**lgsigma_int, ratio_data, cov, n_threads=n_threads)



# =============================================================================
# =============================================================================
# Init

filepath = "/spiff/cwu18/project/massbins/mdpl2"
filename = "halo_morb_fit.h5"
with h5.File(join(filepath, filename), "r") as f:
    halomass = f["fit_morb"][()]

filepath = "/spiff/cwu18/project/massbins/mdpl2"
filename = "galaxy_sm_median.h5"
with h5.File(join(filepath, filename), "r") as f:
    galaxysm = f["median_sm"][()]

path = "/spiff/cwu18/project/halo_mass/mdpl2/fit"
mle_file="mle_full_fixMorb.h5"
mle_name="mle"
with h5.File(join(path, mle_file), "r") as f:
    pars_mle = f["max_posterior"][mle_name][()]

m_pivot = 10**14


DF_mdpl2 = DF(halomass, galaxysm, pars_mle, m_pivot)

# set_subhalomass

path = "/spiff/cwu18/project/dynamical_friction/m_sm"
name = "bestfit_pars.h5"

DF_mdpl2.set_subhalomass(path, name)

# set_factor 
path = "/spiff/cwu18/project/velocity_distribution/mdpl2/cdf/MB_fit"
name = "best_fit_pars_0_1.h5"

DF_mdpl2.set_factor(path, name)


# set_mass_func 
path = "/spiff/cwu18/project/dynamical_friction/mdpl2/mass_func"
name = "mdpl2_mass_func_full.h5"

DF_mdpl2.set_mass_func(path, name)

# set_equs
rh = power_law(halomass / m_pivot, p=pars_mle[0], s=pars_mle[1])

tmax  = 40
t_span = [0, tmax]
y0 = [rh, -1e-10]  
t_eval = np.linspace(0, tmax, 1000) 

DF_mdpl2.set_equs(t_span, y0, t_eval)




# ========================================================================================
# ========================================================================================
# Run chains
alpha = -6
beta = -5
lgsigma_int = -2
pars = [alpha, beta, lgsigma_int]

# load data
n_threads = 25 # parallel threads in solving DF

path ='/spiff/cwu18/project/dynamical_friction/mdpl2/ratio_hghm_binbybin'
with h5.File(join(path, "ratio_smooth.h5"), "r") as f:
    ratio_data = f["ratio"][()]
    cov = f["cov"][()]



# run with a new chain ==============================================================
# path = "/spiff/cwu18/project/dynamical_friction/mcmc/mdpl2"

# chain_file = "chain100.h5"
# chain_name = f"chain"

# run_chain(
#     path=path, 
#     n_steps=100,  n_walkers=6,  n_cpu=6, n_dim= len(pars),
#     log_prob=DF_lkl, 
#     log_prob_args=[ratio_data, cov, n_threads], 
#     pars_init=pars,
#     chain_name=chain_name,
#     chain_file=chain_file,
#     )

# run with a existing chain=========================================================
path = "/spiff/cwu18/project/dynamical_friction/mcmc/mdpl2"

chain_file = "chain110.h5"
chain_name = f"chain"

restart_file = "chain100.h5"
restart_name = f"chain"

run_chain(
    path=path, 
    n_steps=10,  n_walkers=6, n_cpu=6, n_dim= len(pars),
    log_prob=DF_lkl, 
    log_prob_args=[ratio_data, cov, n_threads], 
    restart_file=restart_file,
    restart_name=restart_name,
    overwrite=True,
    chain_name=chain_name,
    chain_file=chain_file,
    )