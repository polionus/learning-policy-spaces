import jax

def get_neighbors(sigma: float, new_indices:jax.Array, elite_population: jax.Array, key: jax.Array):
    mu = elite_population[new_indices]
    population = jax.random.normal(key, shape= mu.shape) + sigma
    return population

def search_sigma_anneal(sigma: float, sigma_min: float, sigma_rate: float):
    sigma = max(sigma_min, sigma * sigma_rate)

def init_population(latent_size: int, population_size: int, key: jax.Array) -> jax.Array:
    return jax.random.normal(key, shape = (population_size, latent_size))
