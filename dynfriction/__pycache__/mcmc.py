import os
# import multiprocessing as mp
# from billiard.pool import Pool
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Tuple, Union

import h5py
import numpy as np
from chainconsumer import ChainConsumer
from emcee.autocorr import integrated_time
from emcee.backends import HDFBackend
from emcee.ensemble import EnsembleSampler
from os.path import join
import shutil

def walker_init(
    pars_init: Union[np.ndarray, List, Tuple],
    log_prob: Callable,
    log_prob_args: Union[List, Tuple],
    n_walkers: int,
    init_scatter: np.ndarray = 0.1,
) -> np.ndarray:
    """Checks initial point for the chain and that all walkers are
    initialized correctly around this point.

    Parameters
    ----------
    pars_init : Union[np.ndarray, List, Tuple]
        Initial point for each parameter. The number of dimensions is inferred 
        from its length 
    log_prob : Callable
        Log probability function
    log_prob_args : Union[List, Tuple]
        Arguments passed to the `log_prob` function
    n_walkers : int
        Number of walkers per dimension
    init_scatter : float, optional, walkers_init = pars_init * (1.0 + init_scatter * rand), rand=[-1, 1)
    Returns
    -------
    np.ndarray
        The initial position for all the walkers. The array has shape 
        (n_walkers, n_dim)

    Raises
    ------
    ValueError
        If log-posterior returned infinity at inital point.
    ValueError
        If log-posterior returned infinity for any of the walkers.
    """
    # Get the number of dimensions (parameters)
    n_dim = len(pars_init)

    # Check initial point.
    lnpost_init = log_prob(pars_init, *log_prob_args)

    if not np.isfinite(lnpost_init):
        raise ValueError("Initial point returned infinity")
    else:
        print(f"\t Initial log posterior {lnpost_init:.2f}")

    # Initialize walkers around initial point with a 10% uniform scatter.
    rand = np.random.uniform(low=-1, size=(n_walkers, n_dim))  # [-1, 1)
    walkers_init = pars_init * (1.0 + init_scatter * rand) 

    # Check walkers.
    lnlike_inits = [log_prob(walkers_init[i], *log_prob_args)
                    for i in range(n_walkers)]

    if not all(np.isfinite(lnlike_inits)):
        raise ValueError("Some walkers are not properly initialized.")

    return walkers_init


def run_chain(
    path: str,
    n_steps: int,
    n_walkers: int,
    n_dim: int,
    log_prob: Callable,
    log_prob_args: Union[List, Tuple],
    pars_init: Union[np.ndarray, List, Tuple] = None,
    restart_file: str = None,
    restart_name: str = 'chain',
    overwrite: bool = False,
    n_cpu: int = 1,
    chain_name: str = 'chain',
    chain_file: str = None,
    init_scatter: np.ndarray = 0.1, 
) -> None:
    # Initialize walkers to the last step if the burn chain.
    if restart_file is None:
        walker_pos = walker_init(pars_init=pars_init, log_prob=log_prob,
                            log_prob_args=log_prob_args, n_walkers=n_walkers, init_scatter=init_scatter)
        # Initialize backend to save chain state
        backend = HDFBackend(join(path, chain_file), name=chain_name)
        backend.reset(n_walkers, n_dim)
    else:
        dest_file = join(path, chain_file)
        src_file = join(path, restart_file)
        if not overwrite and os.path.exists(dest_file):
            raise FileExistsError(f"Destination file {dest_file} already exists.")
        shutil.copy2(src_file, dest_file)
        backend = HDFBackend(dest_file, name=restart_name)
        walker_pos = backend.get_last_sample()
        

        
    with ProcessPoolExecutor(n_cpu) as pool:
        # Instantiate sampler
        sampler = EnsembleSampler(
            nwalkers=n_walkers,
            ndim=n_dim,
            log_prob_fn=log_prob,
            pool=pool,
            backend=backend,
            args=log_prob_args
        )

        sampler.run_mcmc(
            initial_state=walker_pos,
            nsteps=n_steps,
            progress=True,
            progress_kwargs={
                'desc': f"Chain {chain_name}",
                'ncols': 100,
                'colour': 'green'
            }
        )
    return


def summary(
    path: str,
    n_dim: int,
    chain_name: str,   
    mle_name: str,
    chain_file: str = 'chain.h5',
    mle_file: str = 'mle.h5',
    p_labs: list = None,
    burn = 0,
) -> None:
    sampler = HDFBackend(join(path, chain_file), name=chain_name, read_only=True)
    flat_samples = sampler.get_chain(flat=True, discard=int(burn*sampler.iteration))
    log_prob = sampler.get_log_prob(flat=True, discard=int(burn*sampler.iteration))

    tau = [
        integrated_time(flat_samples[:, i], c=150, tol=50)[0]
        for i in range(n_dim)
    ]
    print(f"Autocorrelation length {max(tau):.2f}")

    # Setup chainconsumer for computing MLE parameters
    c = ChainConsumer()
    if p_labs:
        c.add_chain(flat_samples, posterior=log_prob,
                    parameters=p_labs)
    else:
        c.add_chain(flat_samples, posterior=log_prob)
    c.configure(
        summary=False,
        sigmas=[1, 2],
        colors='k',
        # tick_font_size=10,
        # max_ticks=3,
        usetex=True,
    )

    # Compute max posterior and quantiles (16, 50, 84).
    quantiles = np.zeros((n_dim, 3))
    max_posterior = np.zeros(n_dim)
    for i, ((_, v1), (_, v2)) in enumerate(
        zip(
            c.analysis.get_summary().items(),
            c.analysis.get_max_posteriors().items(),
        )
    ):
        quantiles[i, :] = v1
        max_posterior[i] = v2
    cov = c.analysis.get_covariance()[-1]
    corr = c.analysis.get_correlations()[-1]

    mean = np.mean(flat_samples, axis=0)

    # Save to file.
    with h5py.File(join(path, mle_file), 'a') as hdf:
        for ds, var in zip(
            ['quantiles/', 'max_posterior/', 'mean/', 'covariance/', 'correlations/'],
            [quantiles, max_posterior, mean, cov, corr],
        ):
            name_ = ds + mle_name
            # Overwirte existing values.
            if name_ in hdf.keys():
                sp = hdf[name_]
                sp[...] = var
            # Create dataset otherwise.
            else:
                hdf.create_dataset(name_, data=var)
    return None


if __name__ == '__main__':
    pass
