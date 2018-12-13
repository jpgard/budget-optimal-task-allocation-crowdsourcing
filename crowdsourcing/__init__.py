"""
Classes for implementation and simulation with Karger, Oh and Shah (2014):
Budget-Optimal Task Allocation for Reliable Crowdsourcing Systems

Code written by Josh Gardner jpgard@cs.washington.edu
"""
import numpy as np
import time
from scipy.stats import mode


def compute_error_rate(preds, labs):
    """
    Compute error rate between predicted and true labels.
    :param preds: np.array of predictions
    :param labs: np.array of true labels
    :return:
    """
    np.testing.assert_equal(preds.shape, labs.shape)
    error_rate = np.sum(labs != preds) / preds.shape[0]
    return error_rate


class Worker():
    def __init__(self, id, p):
        self.assigned_tasks = []  # list of assigned tasks
        self.p = p  # defines worker reliability
        self.id = id  # defines worker column index in A, G

    def get_noisy_answer(self, task_easiness):
        """
        Fetch an answer to a task; user is correct w.p. p and incorrect w.p. 1-p.
        :return:
        """
        correct_prob = max(0.5,
                           self.p - (1 - task_easiness))  # user pays a penalty of 1-easiness, can't be worse than 0.5
        return np.random.choice([True, False], p=[correct_prob, 1 - correct_prob])


class Task():
    def __init__(self, id, true_label, easiness):
        self.id = id
        assert true_label in [-1, 1]
        self.true_label = true_label
        self.easiness = easiness


class Crowd():
    def __init__(self, m, n, l, r, q, easiness_alpha, easiness_beta, verbose=True):
        self.m = m  # number of tasks
        self.n = n  # number of workers
        self.l = l  # task degree
        self.r = r  # worker degree
        self.q = q
        self.verbose = verbose
        self.easinesss_alpha = easiness_alpha
        self.easiness_beta = easiness_beta
        assert m * l / r == n  # check value of n has correct relation to other parameters
        self.G = np.zeros((self.m, self.n))  # task assignment graph
        self.A = np.zeros((self.m, self.n))  # initalize answer matrix A
        self.workers = []
        self.tasks = []
        self.initialize_workers()
        self.initialize_tasks(m, easiness_alpha=easiness_alpha, easiness_beta=easiness_beta)
        self.initialize_task_assignments(l, r)

    def initialize_workers(self):
        """
        Generate a list of n workers using spammer-hammer model with reliabilities distributed according to q.
        :param n:
        :param f_p:
        :return:
        """
        self.workers = []
        for j in range(self.n):
            # generate p according to spammer-hammer model
            p_j = np.random.choice([1., 0.5], p=[self.q, 1 - self.q])
            worker = Worker(j, p_j)
            self.workers.append(worker)
        return

    def initialize_tasks(self, m, easiness_beta, easiness_alpha):
        """
        Generate a list of m tasks with specified difficulty.
        :param m:
        :param difficulty:
        :return:
        """
        self.tasks = []
        for i in range(m):
            difficulty = np.random.beta(a=easiness_alpha, b=easiness_beta)
            # randomly get task label either -1 or 1
            task_label = 1 if np.random.uniform(low=0.0, high=1.0, size=1) > 0.5 else -1
            task = Task(i, task_label, difficulty)
            self.tasks.append(task)
        return

    def initialize_task_assignments(self, l, r):
        """
        Initialize the task assignment graph G by assigning workers
        :param l:
        :param r:
        :return:
        """
        print("[INFO] initializing task assignments")
        # generate batches of tasks, T_j and assign to workers
        task_ids = [x.id for x in self.tasks]
        task_half_edges = np.random.permutation(np.array([[i] * self.l for i in range(self.m)]).flatten())
        worker_half_edges = np.random.permutation(
            np.array([[j] * self.r for j in range(self.n)]).flatten())  # generate r half-edges for each worker
        for i, j in zip(task_half_edges, worker_half_edges):
            self.G[i, j] += 1
        # print some diagnostics as a sanity check
        print("[INFO] task assignments completed; crowd is ready to work!")
        rowsums = np.sum(self.G, axis=1)
        colsums = np.sum(self.G, axis=0)
        print("[INFO] specified task degree l={}; actual mean task degree ={}".format(self.l,
                                                                                      np.sum(rowsums) / float(self.m)))
        print("[INFO] specified worker task degree r={}; actual mean worker task assignment {}".format(self.r, np.sum(
            colsums) / float(self.n)))
        return

    def generate_task_responses(self):
        """
        Simulate workers responses to their assigned tasks.
        :return:
        """
        print("[INFO] generating task responses")
        e_i, e_j = np.nonzero(self.G)
        for i, j in zip(e_i, e_j):  # iterate over edges in G
            # get worker response
            correct_i_j = self.workers[j].get_noisy_answer(self.tasks[i].easiness)
            a_i_j = self.tasks[i].true_label if correct_i_j else -(self.tasks[i].true_label)
            self.A[i, j] = a_i_j
        return

    def run_kos_inference_algorithm(self, max_num_iters):
        """
        Run the inference algorithm of Karger, Oh and Shah (2014).
        :param max_num_iters: numnber of iterations to perform (k in paper)
        :return:
        """
        print("[INFO] running the inference algorithm for {} iterations".format(max_num_iters))
        e_i, e_j = np.nonzero(self.G)
        for k in range(max_num_iters):
            start = time.time()
            # initialize empty x message array
            x_k = np.full((self.m, self.n), np.nan)
            # initialize entries of Y with N(1,1) random variables
            y_k = np.random.normal(loc=1, scale=1, size=(self.m, self.n))
            # update the task message; represents the log-likelihood of task i being a positive task
            for i, j in zip(e_i, e_j):
                # delta_i_not_j, neighborhood of i excluding j (all workers assigned to task i excluding j)
                delta_i = np.nonzero(self.G[i, :])[0]
                delta_i_not_j = delta_i[delta_i != j]
                x_k[i, j] = np.sum([self.A[i, j_prime] * y_k[i, j_prime] for j_prime in delta_i_not_j])
            # update the worker message; represents how reliable worker j is
            for i, j in zip(e_i, e_j):
                # delta_j_not_i; neighborhood of j excluding i (all tasks assigned to worker j excluding i)
                delta_j = np.nonzero(self.G[:, j])[0]
                delta_j_not_i = delta_j[delta_j != i]
                y_k[i, j] = np.sum([self.A[i_prime, j] * x_k[i_prime, j] for i_prime in delta_j_not_i])
            end = time.time()
            print("[INFO] iteration {} completed in {}s".format(k, round(end - start, 3)))
        # compute final estimates
        x = np.full(self.m, np.nan)
        for i in range(self.m):
            x[i] = np.sum([self.A[i, j] * y_k[i, j] for j in np.nonzero(self.G[i, :])[0]])
        t_hat = np.sign(x)
        return t_hat, y_k

    def run_majority_vote(self):
        """
        Compute estimated labels from existing answers using majority vote.
        :return:
        """
        t_hat = np.full(self.m, np.nan)
        for i in range(self.m):
            try:
                majority_vote_i = mode(self.A[i, :][self.A[i, :] != 0]).mode
                t_hat[i] = majority_vote_i
            except Exception as e:
                print("[WARNING] error computing majority vote for task {}: {}; defaulting to random".format(i, e))
                t_hat[i] = np.random.choice([-1, 1])
        return t_hat

    def run_spectral_estimation(self):
        """
        Use leading left singular vector for label estimation.
        :return:
        """
        u, s, v = np.linalg.svd(self.A)
        t_hat = np.sign(u[:, 0]).astype(int)
        return t_hat


def initialize_and_run_experiment(m, n, l, r, q, iternum, easiness_alpha=50, easiness_beta=1e-6, verbose=True, spectral_error_rate_thresh=0.55, k=28):
    """

    :param m:
    :param n:
    :param l:
    :param r:
    :param q:
    :return:
    """
    # initializes the crowd, including workers, tasks, and task assignments
    print(
        "[INFO] initializing experiment with parameters: m={}; n={}; l={}; r={}; q={}; easiness_alpha {}; easiness_beta {}; iternum={}; k={}".format(m, n, l,
                                                                                                                r, q, easiness_alpha, easiness_beta,
                                                                                                                iternum,
                                                                                                                k))
    crowd = Crowd(m=m, n=n, l=l, r=r, q=q, easiness_alpha=easiness_alpha, easiness_beta=easiness_beta, verbose=verbose)  # q is fixed at 0.3 for this experiment
    crowd.generate_task_responses()
    # now run inference algorithm on results; set k using default value of 28 from paper
    t_hat_kos, y_k = crowd.run_kos_inference_algorithm(max_num_iters=k)
    t_hat_majority_vote = crowd.run_majority_vote()
    t_hat_spectral = crowd.run_spectral_estimation()
    # basic analysis of accuracy; should be roughly loglinear in number of queries per task l
    task_labels = np.array([task.true_label for task in crowd.tasks])
    if compute_error_rate(t_hat_spectral, task_labels) > spectral_error_rate_thresh:
        t_hat_spectral *= -1
    results_row = {"m": m,
                   "n": n,
                   "l": l,
                   "r": r,
                   "q": q,
                   "easiness_alpha": easiness_alpha,
                   "easiness_beta": easiness_beta,
                   "kos": compute_error_rate(t_hat_kos, task_labels),
                   "spectral": compute_error_rate(t_hat_spectral, task_labels),
                   "majority_vote": compute_error_rate(t_hat_majority_vote, task_labels),
                   "iternum": iternum
                   }
    return results_row
