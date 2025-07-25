import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class ChebyshevOutlierDetector:
    """
    A detector for identifying outliers using Chebyshev's Inequality.

    Chebyshev's Inequality:
        For any real-valued random variable X with finite mean μ and finite standard deviation σ,
        the probability that X deviates from its mean by more than k standard deviations is bounded by:
            P(|X - μ| ≥ kσ) ≤ 1 / k²

    This holds for any distribution, not just normal, making it a useful tool for conservative outlier detection.

    Intuition Example:
        Suppose the average score on a test is 70 with a standard deviation of 5.
        Chebyshev says:
            - At most 1/4 (25%) of scores are more than 2 std dev away from the mean.
            - At most 1/9 (11.1%) are more than 3 std dev away.

        This gives you a statistical guarantee — even if you don't know the shape of the score distribution.

    Parameters
    ----------
    data : array-like
        The reference dataset assumed to represent normal behavior.
    filter_low : float, optional
        Lower bound to filter the input data before computing statistics.
    filter_high : float, optional
        Upper bound to filter the input data before computing statistics.
    
    Example usage
    ----------
        example_data = np.random.normal(loc=50, scale=5, size=1000)
        cheb_detector = ChebyshevOutlierDetector(example_data, filter_low=3, filter_high=97)
        test_point = 10
        is_out_cheb = cheb_detector.is_outlier(test_point, k_threshold=3)
        cheb_detector.plot(test_point)
        
    """
    
    def __init__(self, data, filter_low=None, filter_high=None):
        self.original_data = np.array(data)
        self.data = self._apply_filter(self.original_data, filter_low, filter_high)
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)

    def _apply_filter(self, data, low, high):
        if low is not None:
            data = data[data >= low]
        if high is not None:
            data = data[data <= high]
        return data

    def is_outlier(self, test_point, k_threshold=3):
        """
        Check if the test point is an outlier using Chebyshev's inequality.

        Parameters
        ----------
        test_point : float
            The value to evaluate.
        k_threshold : float
            Number of standard deviations to use as the outlier threshold.

        Returns
        -------
        bool
            True if the test point is an outlier (falls beyond k_threshold std dev from mean), False otherwise.
        """
        distance = abs(test_point - self.mean)
        k = distance / self.std if self.std > 0 else np.inf
        return k >= k_threshold

    def plot(self, test_point, k_threshold=3):
        """
        Plot the distribution of the data and show test point along with ±kσ bounds.

        Parameters
        ----------
        test_point : float
            The value to highlight on the plot.
        k_threshold : float
            Number of standard deviations for outlier threshold.
        """
        plt.figure(figsize=(10, 5))
        plt.hist(self.data, bins=30, alpha=0.6, label='Normal Data')

        lower_bound = self.mean - k_threshold * self.std
        upper_bound = self.mean + k_threshold * self.std

        plt.axvline(self.mean, color='green', linestyle='-', label='Mean')
        plt.axvline(lower_bound, color='orange', linestyle='--', label=f'-{k_threshold}σ')
        plt.axvline(upper_bound, color='orange', linestyle='--', label=f'+{k_threshold}σ')
        plt.axvline(test_point, color='red', linestyle='--', label='Test Point')

        plt.title("Chebyshev Inequality Outlier Detection")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class TTestOutlierDetector:
    """
    A detector that uses a t-test to identify whether a test point lies significantly
    in the tail of the distribution (right, left, or both), based on the sample mean and standard deviation.

    Supports plotting in both t-score and original value domains.

    Parameters
    ----------
    data : array-like
        The reference dataset assumed to represent normal behavior.
    filter_low : float, optional
        Lower bound to filter the input data.
    filter_high : float, optional
        Upper bound to filter the input data.
    tail : str
        'right', 'left', or 'two-sided' to define the direction of the test.
    
    Example usage
    ----------
        np.random.seed(42)
        example_data = np.random.normal(loc=0, scale=2, size=50)

        t_detector = TTestOutlierDetector(example_data, tail='two-sided',filter_low=3, filter_high=97)
        test_val = 1.2
        is_out = t_detector.is_outlier(test_val, alpha=0.05)
        t_detector.plot_value(test_val)
        
    """

    def __init__(self, data, filter_low=None, filter_high=None, tail='right'):
        self.original_data = np.array(data)
        self.data = self._apply_filter(self.original_data, filter_low, filter_high)
        self.tail = tail
        self.n = len(self.data)
        self.mean = np.mean(self.data)
        self.std = np.std(self.data, ddof=1)
        self.df = self.n - 1

    def _apply_filter(self, data, low, high):
        if low is not None:
            data = data[data >= low]
        if high is not None:
            data = data[data <= high]
        return data

    def is_outlier(self, test_point, alpha=0.05):
        t_score = (test_point - self.mean) / (self.std / np.sqrt(self.n))

        if self.tail == 'right':
            critical_t = stats.t.ppf(1 - alpha, df=self.df)
            return t_score > critical_t
        elif self.tail == 'left':
            critical_t = stats.t.ppf(alpha, df=self.df)
            return t_score < critical_t
        elif self.tail == 'two-sided':
            critical_t = stats.t.ppf(1 - alpha / 2, df=self.df)
            return abs(t_score) > critical_t
        else:
            raise ValueError("tail must be 'right', 'left', or 'two-sided'")

    def plot_tscore(self, test_point, alpha=0.05):
        t_score = (test_point - self.mean) / (self.std / np.sqrt(self.n))
        x = np.linspace(-5, 5, 500)
        y = stats.t.pdf(x, df=self.df)

        plt.figure(figsize=(10, 5))
        plt.plot(x, y, label='t-distribution (df={})'.format(self.df))

        if self.tail == 'right':
            critical_t = stats.t.ppf(1 - alpha, df=self.df)
            plt.fill_between(x, 0, y, where=x >= critical_t, color='red', alpha=0.3, label='Rejection Region (α)')
        elif self.tail == 'left':
            critical_t = stats.t.ppf(alpha, df=self.df)
            plt.fill_between(x, 0, y, where=x <= critical_t, color='red', alpha=0.3, label='Rejection Region (α)')
        elif self.tail == 'two-sided':
            critical_t = stats.t.ppf(1 - alpha / 2, df=self.df)
            plt.fill_between(x, 0, y, where=np.abs(x) >= critical_t, color='red', alpha=0.3, label='Rejection Region (±α/2)')

        plt.axvline(t_score, color='black', linestyle='--', label=f'Test Point t={t_score:.2f}')
        plt.title(f"{self.tail.title()} t-Test (t-score domain)")
        plt.xlabel("t-score")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_value(self, test_point, alpha=0.05):
        """
        Plot the original value domain, mapping t-distribution back to input space.
        """
        x_vals = np.linspace(self.mean - 5*self.std, self.mean + 5*self.std, 500)
        t_scores = (x_vals - self.mean) / (self.std / np.sqrt(self.n))
        y = stats.t.pdf(t_scores, df=self.df)

        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, y, label='t-distribution (mapped to original values)')

        if self.tail == 'right':
            critical_t = stats.t.ppf(1 - alpha, df=self.df)
            critical_val = self.mean + critical_t * (self.std / np.sqrt(self.n))
            plt.fill_between(x_vals, 0, y, where=x_vals >= critical_val, color='red', alpha=0.3, label='Rejection Region (α)')
        elif self.tail == 'left':
            critical_t = stats.t.ppf(alpha, df=self.df)
            critical_val = self.mean + critical_t * (self.std / np.sqrt(self.n))
            plt.fill_between(x_vals, 0, y, where=x_vals <= critical_val, color='red', alpha=0.3, label='Rejection Region (α)')
        elif self.tail == 'two-sided':
            critical_t = stats.t.ppf(1 - alpha / 2, df=self.df)
            upper_val = self.mean + critical_t * (self.std / np.sqrt(self.n))
            lower_val = self.mean - critical_t * (self.std / np.sqrt(self.n))
            plt.fill_between(x_vals, 0, y, where=(x_vals >= upper_val) | (x_vals <= lower_val),
                             color='red', alpha=0.3, label='Rejection Region (±α/2)')

        plt.axvline(test_point, color='black', linestyle='--', label=f'Test Point = {test_point:.2f}')
        plt.title(f"{self.tail.title()} t-Test (Original Value Domain)")
        plt.xlabel("Original Value")
        plt.ylabel("Mapped t-distribution PDF")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":

    # Test usage for value-based plotting
    np.random.seed(42)
    example_data = np.random.normal(loc=0, scale=2, size=50)

    t_detector = TTestOutlierDetector(example_data, tail='two-sided',filter_low=3, filter_high=97)
    test_val = 1.2
    is_out = t_detector.is_outlier(test_val, alpha=0.05)
    t_detector.plot_value(test_val)