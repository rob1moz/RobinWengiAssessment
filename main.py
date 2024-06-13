import numpy as np
import pandas as pd
import seaborn as sns


class MonteCarlo:
    def __init__(self, inputs):
        # Define the value of the seed
        np.random.seed(inputs['random_seed_value'])
        # Define the frequency random vector of the size of the number of simulated periods
        self.frequency_vector = self.get_random_vector(inputs['frequency_distribution'],
                                                       inputs['frequency_parameter'],
                                                       inputs['number_of_simulation'])
        # Define the severity random vector of the size of the total frequency
        self.severity_vector = self.get_random_vector(inputs['severity_distribution'],
                                                      inputs['severity_parameter'],
                                                      np.sum(self.frequency_vector))

    def get_random_vector(self, a_distribution, a_parameter_set, a_number_of_simulation):
        # Generate a random vector from a distribution given by a string.
        if a_distribution == 'poisson':
            random_vector = np.random.poisson(a_parameter_set['lambda'], a_number_of_simulation)
        elif a_distribution == 'pareto':
            random_vector = (np.random.pareto(a_parameter_set['alpha'], a_number_of_simulation) + 1) * \
                            a_parameter_set['theta']
        else:
            print('The selected distribution in not yet implemented')
        return random_vector
    
    def plot_frequency(self):
        # Plot the Frequency random vector
        return sns.displot(self.frequency_vector, bins=5, kde=True,kde_kws={"bw_adjust": 5})
    
    def plot_severity(self):
        # Plot the severity random vector
        return sns.displot(self.severity_vector, bins=100, kde=True,kde_kws={"bw_adjust": 0.5},log_scale=True)


class XlContract:
    def __init__(self, inputs):
        # Store the inputs in an attribute
        self.inputs = inputs
        # Instantiate the monte carlo simulation
        self.monte_carlo = MonteCarlo(self.inputs)
        # 1. loss vector for USD 10m xs USD 5m
        self.xs_loss = self.get_xs_loss_vector()
        # 2. loss vector for 10m xs 5m, AAD 2m
        self.aad_loss = self.get_loss_vector_with_aad_and_aal(self.xs_loss, self.inputs['AAD'], np.inf)
        # 3. loss vector for 10m xs 5m, AAL 12m
        self.aal_loss = self.get_loss_vector_with_aad_and_aal(self.xs_loss,0, self.inputs['AAL'])
        # 4. loss vector for 10m xs 5m, AAD 2m, AAL 12m
        self.aad_aal_loss = self.get_loss_vector_with_aad_and_aal(self.xs_loss,self.inputs['AAD'],self.inputs['AAL'])
        # Create a DataFrame with all relevant result for each contract from 1. to 4.
        self.summary = self.get_result_summary()

    def get_xs_loss_vector(self):
        # Define the cumulative claim pattern from the primary insurer
        primary_pattern = np.ones((1,len(self.inputs['claim_pattern']))) * self.inputs['claim_pattern']
        # Define the cumulative individual claim according to the claim pattern
        cumulative_individual_claim = self.monte_carlo.severity_vector[..., np.newaxis] * np.cumsum(primary_pattern)

        # Apply the limit and deductible to individual claims and get a vector of severity under xs conditions
        individual_claims_with_xs = np.maximum(np.minimum(
            self.inputs['limit'], cumulative_individual_claim - self.inputs['deductible']), 0)
        individual_claims_with_xs[..., 1:] = np.diff(individual_claims_with_xs)

        # Allocate the individual_claims_with_xs to the simulated periods according to the frequency of each period
        xs_loss = np.zeros((self.inputs['number_of_simulation'], individual_claims_with_xs.shape[1]))
        for i in range(individual_claims_with_xs.shape[1]):
            xs_loss[..., i] = self.allocate_claim_to_simulation_period(individual_claims_with_xs[..., i])
        return xs_loss

    def get_loss_vector_with_aad_and_aal(self, a_loss_vector, aad, aal):
        # Apply the generic formula for AAD and AAL limit on aggregated claims
        aad_aal_loss = np.minimum(np.maximum(np.cumsum(a_loss_vector, axis=1) - aad, 0), aal)
        # Correct the cumulative for the claim pattern
        aad_aal_loss[..., 1:] = np.diff(aad_aal_loss)
        return aad_aal_loss

    def allocate_claim_to_simulation_period(self, an_individual_claim_vector):
        # We want to aggregate the individual claim according to the frequency
        # the goal is to have a claim_vector that match the simulation periods
        claim_frequency = self.monte_carlo.frequency_vector
        relevant_interval = np.cumsum(claim_frequency)
        # Define the aggregated claim vector with same shape as the frequency
        claim_vector = np.zeros_like(claim_frequency, dtype=float)
        # Sum the correct number of claim for each simulation period taking the claims in sequential order
        claim_vector[claim_frequency > 0] = [
            an_individual_claim_vector[
            relevant_interval[i] - claim_frequency[i]:relevant_interval[i]].sum()
            for i in range(len(claim_frequency)) if claim_frequency[i] > 0]
        return claim_vector

    def VaR(self, a_loss_vector, a_level):
        # Value at risk
        return np.quantile(a_loss_vector, a_level)

    def TVaR(self, a_loss_vector, a_level):
        # Tail Value at risk
        return a_loss_vector[a_loss_vector >= self.VaR(a_loss_vector, a_level)].mean()

    def premium(self,a_loss_vector, a_level):
        # Premium according to pricing model
        mean = a_loss_vector.mean()
        tvar = self.TVaR(a_loss_vector,a_level)
        cost_of_capital = self.inputs['cost_of_capital']
        return mean + (tvar-mean) * cost_of_capital

    def reinsurer_pattern(self, a_loss_array):
        # Compute the reinsurer payment pattern based on losses
        # Format the ouput as a string representation of a list with of 3 digits number inside
        pattern_list = list(a_loss_array.sum(axis=0)/a_loss_array.sum())
        pattern_list_formatted = ['%.3f' % elem for elem in pattern_list]
        return str(pattern_list_formatted)

    def get_contract_results(self, a_loss_array):
        # Compute the result summary for a particular contract kind represented by a_loss_array
        # Reduce the dimension of the Pattern first then store the results in a dictionary
        a_loss_vector = a_loss_array.sum(axis=1)
        contract_result_summary = {
            'Average Loss': a_loss_vector.mean(),
            'VaR_99': self.VaR(a_loss_vector,0.99),
            'TVaR_99': self.TVaR(a_loss_vector,0.99),
            'VaR_995': self.VaR(a_loss_vector,0.995),
            'Premium': self.premium(a_loss_vector, 0.99),
            'Pattern': self.reinsurer_pattern(a_loss_array)
        }
        return contract_result_summary

    def get_result_summary(self):
        # Store the results for each contract type in a dictionary of dictionary and convert it in a DataFrame
        summary ={
            '10m xs 5m': self.get_contract_results(self.xs_loss),
            '10m xs 5m, AAD 2m': self.get_contract_results(self.aad_loss),
            '10m xs 5m, AAL 12m': self.get_contract_results(self.aal_loss),
            '10m xs 5m, AAD 2m, AAL 12m': self.get_contract_results(self.aad_aal_loss)
        }
        return pd.DataFrame(summary)

    def the_basic_challenge(self):
        # Return the results for the first question about the basic challenge as a Serie
        return self.get_result_summary().loc[['Average Loss','VaR_99'],'10m xs 5m']

    def introducing_AAD_and_AAL(self):
        # Return the results for the second question about the AAD and AAL thresholds as a DataFrame
        return self.get_result_summary().loc[['Average Loss','VaR_99']]

    def applying_a_simple_pricing_model(self):
        # Return the results for the pricing model as a DataFrame
        return self.get_result_summary().loc[['Premium']]

    def patterns(self):
        # Return the results of the reinsurer pattern in a DataFrame
        return self.get_result_summary().loc[['Pattern']].T


class Portfolio(XlContract):
    def __init__(self, input_list):
        # Store the inputs in an attribute the input_list is a list of dictionary
        self.input_list = input_list
        # Define a dictionary with all the line of business objects
        self.lob = self.get_all_lob()

    def get_all_lob(self):
        # Build a dictionary of with Ligne of business name and Xl_contract object instantiated
        # We loop over input_list to generate all lignes of business
        lob = {}
        for i in self.input_list:
            lob[i['LOB']] = XlContract(i)
        return lob

    def get_portfolio_underwriting_risk(self, a_loss_kind):
        # Compute the Underwriting risk random variable for the all portfolio
        # We define this risk as the difference between the losses and the premiums
        # The risk is aggregated for all ligne of business
        premiums = []
        losses = []
        for v in self.lob.values():
            a_lob_loss = getattr(v, a_loss_kind).sum(axis=1)
            premiums.append(v.premium(a_lob_loss, 0.99))
            losses.append(a_lob_loss)
        return sum(losses) - sum(premiums)

    def get_portfolio_results(self, a_loss_kind):
        # Compute the SST and Solvency II risk measure for the underwriting risk at portfolio level
        # a_loss_kind refers to a single contract type as define in the exercice
        a_loss_vector = self.get_portfolio_underwriting_risk(a_loss_kind)
        contract_result_summary = {
            'TVaR_99': self.TVaR(a_loss_vector,0.99),
            'VaR_995': self.VaR(a_loss_vector,0.995)
        }
        return contract_result_summary

    def get_portfolio_summary(self):
        # Store the results for each contract type in a dictionary of dictionary and convert it in a DataFrame
        summary = {
            '10m xs 5m': self.get_portfolio_results('xs_loss'),
            '10m xs 5m, AAD 2m': self.get_portfolio_results('aad_loss'),
            '10m xs 5m, AAL 12m': self.get_portfolio_results('aal_loss'),
            '10m xs 5m, AAD 2m, AAL 12m': self.get_portfolio_results('aad_aal_loss')
        }
        return pd.DataFrame(summary)