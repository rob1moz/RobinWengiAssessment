import numpy as np


# If you want predictable result than set the seed to 0. To generate new random numbers then comment the line below
np.random.seed(0)

number_of_simulation = 100000

cost_of_capital = 0.08

## Modeling claim frequency
lam = 1
claim_frequency = np.random.poisson(lam,number_of_simulation)
## Modeling claim severity
alpha = 1.2
theta = 2
claim_severity = (np.random.pareto(alpha, np.sum(claim_frequency)) + 1) * theta


## Determine the losses with XL contract "USD 10m xs USD 5m"
limit = 10
deductible = 5

# Apply the limit and deductible to individual claims
claim_severity_with_xs = np.maximum(np.minimum(limit,claim_severity-deductible),0)

# Determine the total loss per simulated period
xs_loss = np.zeros_like(claim_frequency, dtype=float)
xs_loss[claim_frequency > 0] = [claim_severity_with_xs[np.cumsum(claim_frequency)[i] - claim_frequency[i]:np.cumsum(claim_frequency)[i]].sum() for i in range(len(claim_frequency)) if claim_frequency[i] > 0]


xs_mean_loss = xs_loss.mean()
xs_VaR_99 = np.quantile(xs_loss,0.99)

xs_TVar_99 = xs_loss[xs_loss>xs_VaR_99].mean()

xs_premium = xs_mean_loss + (xs_TVar_99-xs_mean_loss) * cost_of_capital

## Determine the losses under the Xl with AAD contract "10m xs 5m, AAD 2m"
aad = 2
aad_xs_loss = np.maximum(xs_loss - aad,0)

aad_xs_mean_loss = np.average(aad_xs_loss)
aad_xs_VaR_99 = np.quantile(aad_xs_loss,0.99)
## Determine the losses under the Xl with AAL contract "10m xs 5m, AAL 12m"
aal = 12
aal_xs_loss = np.minimum(xs_loss,aal)

## Determine the losses under the Xl with AAL contract "10m xs 5m, AAD 2m, AAL 12m"

aal_aad_xs_loss = np.minimum(np.maximum(xs_loss - aad,0),aal)


## Patterns

y1 = 0.3
y2 = 0.6
y3 = 0.1

loss1 = claim_severity*y1
loss2 = claim_severity*y2
loss3 = claim_severity*y3

xs_loss1 = np.maximum(np.minimum(limit,loss1-deductible),0)
xs_loss2 = np.maximum(np.minimum(limit,loss1+loss2-deductible)-xs_loss1,0)
xs_loss3 = np.maximum(np.minimum(limit,loss1+loss2+loss3-deductible)-xs_loss1-xs_loss2,0)

xs_y1 = xs_loss1.sum()/xs_loss.sum()
xs_y2 = xs_loss2.sum()/xs_loss.sum()
xs_y3 = xs_loss3.sum()/xs_loss.sum()