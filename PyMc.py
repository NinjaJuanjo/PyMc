import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Datos
n = 20
aprobados = 14

# Modelo bayesiano
with pm.Model() as modelo_aprobacion:
    
    # Prior (distribución previa)
    p = pm.Beta("p", alpha=1, beta=1)
    
    # Likelihood (modelo de los datos)
    obs = pm.Binomial("obs", n=n, p=p, observed=aprobados)
    
    # Muestreo MCMC
    trace = pm.sample(2000, return_inferencedata=True)

# Resumen estadístico
print(az.summary(trace))

# Gráfica de la distribución posterior
az.plot_posterior(trace)
plt.show()