import numpy as np
import matplotlib.pyplot as plt
import rustworkx as rx


class Annealer():
    """
    Simulated Annealing for both binary and continuous problems.

    A variety of options to control the anneal schedule, aswell as custom
    methods for common problems.

    Methods:
        _generate_new_solution: Upon rejection of a potential solution,
            generate a new one.
        anneal: A general annealing function, with options for discrete data.
        maxcut: The weighted MaxCut graph problem.
        portfolio_optimisation: The Markowitz Portfolio poroblem.
    """

    def __init__(self, max_temp=20, min_temp=0.001, min_energy=-10000,
                 alpha=0.99, step_size=0.5):
        """
        Initialise Annealer with global parameters.

        Args:
            max_temp   (float): Initial system temperature.
            min_temp   (float): Stopping Temperature.
            min_energy (float): Stopping energy.
            alpha      (float): Temperature decay rate. In the range (0, 1).
            step_size  (float): New solution deviation amount. For
                _generate_new_solution()
        """
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.min_energy = min_energy
        self.alpha = alpha
        self.step_size = step_size

    def plot_schedule(self):
        """
        Plot the temperature decay for the internal parameters.
        """
        x_max = np.log(self.min_temp / self.max_temp) / np.log(self.alpha)
        x = np.linspace(0, x_max, 100)
        plt.plot(x, self.max_temp * (self.alpha) ** x)
        plt.title(f"Temperature Decay - alpha={self.alpha}")
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.show()

    def _generate_new_solution(self, x, T, binary, step_size):
        """
        Upon rejection of a potential solution, generate a new one.

        Args:
            x         (array): Current solution guess.
            T         (float): Current temperature.
            binary  (boolean): Whether a binary problem is being solved.
            step_size (float): Deviation amount for new solution.

        Returns:
            array: New guess for soluiton.
        """
        if binary:
            neighbor = x[:]
            index = np.random.randint(0, len(x))
            neighbor[index] = 1 - neighbor[index]
            return neighbor
        else:
            perturbation = np.random.normal(loc=0.0, scale=step_size * T,
                                            size=x.shape)
            return x + perturbation

    def anneal(self, f, x0, binary=False):
        """
        Perform simulated annealing.

        Args:
            f     (function): Objective function.
            x0       (array): Initial guess.
            binary (boolean): Whether a binary problem is being solved.

        Returns:
            array: Guess at solution to problem.
            array: All guesses throughout anneal.
            array: The energies of the problem at each iteration.

        """
        if binary:
            if not all(xi in (0, 1) for xi in x0):
                raise ValueError(f'With binary=True, initial guess x0 must \
                                  have all entries equal to 1 or 0. This \
                                  is not the case for x0 = {x0}')

        T = self.max_temp
        x = x0
        E = f(x0)

        xs = [x0]
        Es = [E]

        while (T > self.min_temp) and (E > self.min_energy):
            x_new = self._generate_new_solution(x, T, binary, self.step_size)
            E_new = f(x_new)
            delE = E_new - E
            if delE < 0 or np.random.rand() < np.exp(-delE / T):
                x = x_new
                E = E_new
            xs.append(x_new)
            Es.append(E_new)
            T *= self.alpha

        return x, xs, Es

    def maxcut(self, edges, x0=None, draw=False, edge_labels=False):
        """
        The weighted MaxCut graph problem.

        Args:
            edges          (list): Tuples representing the edges of a graph,
                with correspongind weights.
            x0            (array): Initial guess of a cut.
            draw        (boolean): Draw the solution.
            edge_labels (boolean): add weights to the visualisation.

        Returns:
            array: Guess at solution to problem.
            array: All guesses throughout anneal.
            array: The energies of the problem at each iteration.
        """
        max_node = max(max(edge[:-1]) for edge in edges) + 1
        if x0 is None:
            x0 = np.random.choice([0, 1], size=max_node)

        def f(x):
            tot = 0
            for edge in edges:
                i, j, w = edge
                tot += w * (x[i] + x[j] - 2*x[i]*x[j])
            return -tot

        solution, guesses, energies = self.anneal(f, x0, binary=True)

        if solution[0] == 1:  # Normalise Solution
            solution = 1 - solution

        if draw:
            graph = rx.PyGraph()
            graph.add_nodes_from(np.arange(0, max_node, 1))
            graph.add_edges_from(edges)
            colors = ["tab:blue" if i == 0 else "tab:red" for i in solution]
            pos, _ = rx.spring_layout(graph), plt.axes(frameon=True)
            rx.visualization.mpl_draw(graph, node_color=colors, node_size=200,
                                      alpha=0.8, pos=pos,
                                      edge_labels=str if edge_labels else None,
                                      with_labels=True, font_weight='bold')
            plt.show()

        return solution, guesses, energies

    def portfolio_optimisation(self, data, budget=None, penalty=100,
                               q=1, x0=None):
        """
        The Markowitz Portfolio Optimisation problem.

        Args:
            data (DataFrame): Historic proces of a selection of stocks.
            budget (Boolean): Option for a budget constraint.
            penalty  (float): Penalty term for optional budget constraint.
            q        (float): Risk factor.
            x0       (array): Initial guess of solution.

            Returns:
                array: Guess at solution to problem.
                array: All guesses throughout anneal.
                array: The energies of the problem at each iteration.
        """
        mu = data.mean().to_numpy()
        sigma = data.cov().to_numpy()
        if x0 is None:
            x0 = np.random.choice([0, 1], size=len(mu))

        if budget:
            def f(x):
                return q * x.T @ sigma @ x - mu.T @ x + \
                       penalty * (budget - np.sum(x)) ** 2
        else:
            def f(x):
                return q * x.T @ sigma @ x - mu.T @ x

        return self.anneal(f, x0, binary=True)
