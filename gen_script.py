"""
    Author: Jose Manuel Bustarviejo Casado (100013022@alumnos.uimp.es)

    Date: 11/08/2025

    Subject: 102778 - Machine learning and quantum computers // Machine learning y ordenadores cuánticos

    Description: This script generates random data from different distributions. Compare:
    (a) a normal or Gaussian distribution for different values of the variance and mean,
    (b) a uniformly random distribution,
    (c) the beta distribution.


"""

import numpy as np
import matplotlib.pyplot as plt


class DataRandomGen:
    """
    This class generates the random datasets for Normal, Uniform Random and Beta distributions
    """

    def __init__(self, n_samples):
        """
        Class constructor
        :param n_samples: Number of data points to generate for each distribution
        """
        self.N_SAMPLES = n_samples

    def normal_g_dist(self, mean, variance):
        """
        Normal (Gaussian) Distribution
        :param mean: Mean parameter
        :param variance: Variance parameter
        :return: Data for Normal (Gaussian) Distribution
        """
        data_normal = np.random.normal(mean, variance, self.N_SAMPLES)
        return data_normal

    def uniform_random_dist(self, low_bound, high_bound):
        """
        Uniform Random Distribution
        :param low_bound: Low bound for Uniform Random Distribution interval
        :param high_bound: High bound for Uniform Random Distribution interval
        :return: Data for Uniform Random Distribution
        """
        data_uniform = np.random.uniform(low_bound, high_bound, self.N_SAMPLES)
        return data_uniform

    def beta_dist(self, alpha, beta):
        """
        Beta Distribution: it is defined on the interval [0, 1]
        :param alpha: Alpha parameter
        :param beta: Beta parameter
        :return: Data for Beta Distribution
        """
        data_beta_dist = np.random.beta(alpha, beta, self.N_SAMPLES)
        return data_beta_dist


class DataPlot:
    """
    This class subplots the data for each distribution
    """

    def __int__(self):
        """
        Class constructor
        """
        pass

    @staticmethod
    def plot_normal_g_dist_subplot(ax, data_n1, mu_1, sigma_1,
                                   data_n2, mu_2, sigma_2, data_n3,
                                   mu_3, sigma_3, n_bins):
        """
        Printing only the axis for the Normal (Gaussian) Distribution subplot, making it more modular
        :param ax: axis object
        :param data_n1: dataset for 1. Standard Normal: Mean = 0, Variance = 1
        :param mu_1: mean value for 1. Standard Normal: Mean = 0, Variance = 1
        :param sigma_1: variance value for 1. Standard Normal: Mean = 0, Variance = 1
        :param data_n2: dataset for 2. Shifted and Narrower: Mean = 5, Variance = 0.5 (Std Dev = sqrt(0.5) ≈ 0.707)
        :param mu_2: mean value for 2. Shifted and Narrower: Mean = 5, Variance = 0.5 (Std Dev = sqrt(0.5) ≈ 0.707)
        :param sigma_2: variance value for 2. Shifted and Narrower: Mean = 5,
        Variance = 0.5 (Std Dev = sqrt(0.5) ≈ 0.707)
        :param data_n3: dataset for 3. Wider: Mean = 0, Variance = 4 (Std Dev = 2)
        :param mu_3: mean value for 3. Wider: Mean = 0, Variance = 4 (Std Dev = 2)
        :param sigma_3: variance value for 3. Wider: Mean = 0, Variance = 4 (Std Dev = 2)
        :param n_bins: Number of bins for the histograms
        """
        ax.hist(data_n1, bins=n_bins, alpha=0.6, label=f'Normal ($\\mu$={mu_1}, $\\sigma^2$={sigma_1 ** 2})')
        ax.hist(data_n2, bins=n_bins, alpha=0.6, label=f'Normal ($\\mu$={mu_2}, $\\sigma^2$={sigma_2 ** 2:.1f})')
        ax.hist(data_n3, bins=n_bins, alpha=0.6, label=f'Normal ($\\mu$={mu_3}, $\\sigma^2$={sigma_3 ** 2})')
        ax.set_title('(a) Normal (Gaussian) Distributions')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.5)

    @staticmethod
    def plot_uniform_random_dist_subplot(ax, data_uni, low_uni, high_uni, n_bins):
        """
        Prints the subplot for the Uniform Random distribution
        :param ax: axis object
        :param data_uni: dataset for Uniform Random distribution
        :param low_uni: low bound for the Uniform Random distribution interval
        :param high_uni: high bound for the Uniform Random distribution interval
        :param n_bins: Number of bins for the histograms
        """
        ax.hist(data_uni, bins=n_bins, color='orange', edgecolor='black', alpha=0.8,
                label=f'Uniform [{low_uni}, {high_uni})')
        ax.set_title('(b) Uniform Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.5)

    @staticmethod
    def plot_beta_subplot(ax, data_b1, a_b1, b_b1, data_b2, a_b2, b_b2, data_b3, a_b3, b_b3, n_bins):
        """
        Prints the subplot for the Beta distribution
        :param ax: axis object
        :param data_b1: dataset for 1. Symmetric (a=b=2) - Bell-shaped
        :param a_b1: alpha value for 1. Symmetric (a=b=2) - Bell-shaped
        :param b_b1: beta value for 1. Symmetric (a=b=2) - Bell-shaped
        :param data_b2: dataset for 2. Skewed Left (a < b) - More weight on the right
        :param a_b2: alpha value for 2. Skewed Left (a < b) - More weight on the right
        :param b_b2: beta value for 2. Skewed Left (a < b) - More weight on the right
        :param data_b3: dataset for 3. Skewed Right (a > b) - More weight on the left
        :param a_b3: alpha value for 3. Skewed Right (a > b) - More weight on the left
        :param b_b3: beta value for 3. Skewed Right (a > b) - More weight on the left
        :param n_bins: Number of bins for the histograms
        """
        ax.hist(data_b1, bins=n_bins, alpha=0.6, label=f'Beta (a={a_b1}, b={b_b1}) - Symmetric')
        ax.hist(data_b2, bins=n_bins, alpha=0.6, label=f'Beta (a={a_b2}, b={b_b2}) - Skewed Left')
        ax.hist(data_b3, bins=n_bins, alpha=0.6, label=f'Beta (a={a_b3}, b={b_b3}) - Skewed Right')
        ax.set_title('(c) Beta Distributions')
        ax.set_xlabel('Value (0 to 1)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.5)

    @staticmethod
    def plot_comparison_subplot(ax, data_n1, data_uni, data_b1, n_bins):
        """
        Prints the comparison subplot
        :param ax: axis object
        :param data_n1: dataset for the Normal (Gaussian) Distribution
        :param data_uni: dataset for the Uniform Random distribution
        :param data_b1: dataset for the Beta distribution
        :param n_bins: Number of bins for the histograms
        """
        ax.hist(data_n1, bins=n_bins, histtype='step', linewidth=2, label='Normal (Standard)')
        ax.hist(data_uni, bins=n_bins, histtype='step', linewidth=2, label='Uniform')
        ax.hist(data_b1, bins=n_bins, histtype='step', linewidth=2, label='Beta (Symmetric)')
        ax.set_title('All Distributions (Comparison)')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.5)

    def create_comparison_figure(self, data_dict, param_dict, n_bins):
        """
        This function manages the structure creation (Figure & Axes), call the modular functions for printing, apply
        final settings and return the Figure object with the comparison subplots
        :param data_dict: dictionary which contains the datasets used for plotting the distributions
        :param param_dict: dictionary which contains the parameters used by each distribution
        :param n_bins: Number of bins for the histograms
        :return: Final figure with comparison subplots
        """
        # 1. Create Figure and Axes (Figure and Axes)
        fig, ax_array = plt.subplots(2, 2, figsize=(15, 10))

        fig.suptitle('Comparison of Different Probability Distributions', fontsize=16)

        # 2. Call each modular functions (ax)

        # Subplot (a) - Row 0, Column 0
        self.plot_normal_g_dist_subplot(
            ax=ax_array[0, 0],
            data_n1=data_dict['normal_1'], mu_1=param_dict['mu_1'], sigma_1=param_dict['sigma_1'],
            data_n2=data_dict['normal_2'], mu_2=param_dict['mu_2'], sigma_2=param_dict['sigma_2'],
            data_n3=data_dict['normal_3'], mu_3=param_dict['mu_3'], sigma_3=param_dict['sigma_3'],
            n_bins=n_bins
        )

        # Subplot (b) - Row 0, Column 1
        self.plot_uniform_random_dist_subplot(
            ax=ax_array[0, 1],
            data_uni=data_dict['uniform'], low_uni=param_dict['low_uni'], high_uni=param_dict['high_uni'],
            n_bins=n_bins
        )

        # Subplot (c) - Row 1, Column 0
        self.plot_beta_subplot(
            ax=ax_array[1, 0],
            data_b1=data_dict['beta_1'], a_b1=param_dict['a_beta_1'], b_b1=param_dict['b_beta_1'],
            data_b2=data_dict['beta_2'], a_b2=param_dict['a_beta_2'], b_b2=param_dict['b_beta_2'],
            data_b3=data_dict['beta_3'], a_b3=param_dict['a_beta_3'], b_b3=param_dict['b_beta_3'],
            n_bins=n_bins
        )

        # Subplot (d) - Row 1, Column 1
        self.plot_comparison_subplot(
            ax=ax_array[1, 1],
            data_n1=data_dict['normal_1'], data_uni=data_dict['uniform'], data_b1=data_dict['beta_1'],
            n_bins=n_bins
        )

        # 3. Final adjustment
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 4. Return the figure
        return fig


if __name__ == '__main__':
    N_BINS = 50

    # Normal parameters
    mu_n_1, sigma_n_1 = 0, 1
    mu_n_2, sigma_n_2 = 5, np.sqrt(0.5)
    mu_n_3, sigma_n_3 = 0, 2

    # Uniform parameters
    low_uni_1, high_uni_1 = 0, 10

    # Beta parameters
    a_beta_01, b_beta_01 = 2, 2
    a_beta_02, b_beta_02 = 5, 1
    a_beta_03, b_beta_03 = 1, 5

    # Instancing the DataRandomGen class
    DRG = DataRandomGen(10000)

    """
    Next, it is going to insert different mean and variance values for generating different
    Normal (Gaussian) Distribution datasets
    """
    # 1. Standard Normal: Mean = 0, Variance = 1
    data_normal_01 = DRG.normal_g_dist(mu_n_1, sigma_n_1)
    # 2. Shifted and Narrower: Mean = 5, Variance = 0.5 (Std Dev = sqrt(0.5) ≈ 0.707)
    data_normal_02 = DRG.normal_g_dist(mu_n_2, sigma_n_2)
    # 3. Wider: Mean = 0, Variance = 4 (Std Dev = 2)
    data_normal_03 = DRG.normal_g_dist(mu_n_3, sigma_n_3)

    """
    Now, it is going to generate the dataset for Uniform Random Distribution
    """
    # Range: [low, high) -> [0, 10)
    data_uniform_01 = DRG.uniform_random_dist(low_uni_1, high_uni_1)

    """
    Finally, it is going to generate the datasets for Beta Distribution
    """
    # 1. Symmetric (a=b=2) - Bell-shaped
    data_beta_01 = DRG.beta_dist(a_beta_01, b_beta_01)
    # 2. Skewed Left (a < b) - More weight on the right
    data_beta_02 = DRG.beta_dist(a_beta_02, b_beta_02)
    # 3. Skewed Right (a > b) - More weight on the left
    data_beta_03 = DRG.beta_dist(a_beta_03, b_beta_03)

    data_dict_1 = {
        'normal_1': data_normal_01,
        'normal_2': data_normal_02,
        'normal_3': data_normal_03,
        'uniform': data_uniform_01,
        'beta_1': data_beta_01,
        'beta_2': data_beta_02,
        'beta_3': data_beta_03,
    }
    param_dict_1 = {
        'mu_1': mu_n_1, 'sigma_1': sigma_n_1, 'mu_2': mu_n_2, 'sigma_2': sigma_n_2, 'mu_3': mu_n_3,
        'sigma_3': sigma_n_3,
        'low_uni': low_uni_1, 'high_uni': high_uni_1,
        'a_beta_1': a_beta_01, 'b_beta_1': b_beta_01, 'a_beta_2': a_beta_02, 'b_beta_2': b_beta_02,
        'a_beta_3': a_beta_03, 'b_beta_3': b_beta_03
    }

    dp = DataPlot()

    # Getting the final figure representation
    final_figure = dp.create_comparison_figure(data_dict_1, param_dict_1, N_BINS)

    # Save figure in a picture:
    final_figure.savefig('compare-distributions.png', dpi=300)

    # Showing the final figure
    plt.show()
