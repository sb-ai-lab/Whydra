import numpy as np

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - informative error path
    raise ImportError(
        "TensorFlow is required for the NOTEARS implementation. "
        "Please install the dependencies listed in requirements.txt."
    ) from exc


# NOTEARS implementation relies on TF1-style graphs (placeholders, sessions).
# We keep compatibility with TF2 by enabling the v1 API explicitly.
tf.compat.v1.disable_eager_execution()


class NoTearTF:
    """
    TensorFlow implementation of the NOTEARS optimization routine.

    This class mirrors the original reference implementation:
    https://github.com/howchihlee/BNGPU/blob/master/BNGPU/NOTEARS.py
    """

    def __init__(self, lamb=0.05, alpha_init=0.0, rho_init=1.0, rho_multiplier=1.25):
        self.alpha = alpha_init
        self.rho = rho_init
        self.lamb = lamb
        self.rho_multiplier = rho_multiplier
        self.graph_nodes = {}

    def construct_graph(self, X: np.ndarray):
        """
        Builds the TF computation graph for the NOTEARS objective.

        Args:
            X: Observational data of shape (n_samples, n_features).
        """
        n, d = X.shape
        graph = tf.Graph()

        with graph.as_default():
            X_tf = tf.constant(X.astype(np.float32), name="sem_data")
            W_var = tf.compat.v1.get_variable(
                "W",
                shape=(d, d),
                initializer=tf.compat.v1.random_normal_initializer(stddev=0.01),
            )
            W_tf = tf.linalg.set_diag(W_var, tf.zeros(d))

            rho_tf = tf.compat.v1.placeholder(tf.float32, name="rho")
            lambda_tf = tf.compat.v1.placeholder(tf.float32, name="lambda")
            d_lrd = tf.compat.v1.placeholder(tf.float32, name="opt_step")
            alpha_tf = tf.compat.v1.placeholder(tf.float32, name="alpha")

            d_xw = X_tf - tf.matmul(X_tf, W_tf)

            loss_penalty = tf.linalg.trace(tf.linalg.expm(tf.square(W_tf))) - float(d)

            loss_xw = tf.reduce_sum(tf.square(d_xw)) / (float(n) * 2.0)
            loss_obj = loss_xw + (rho_tf / 2.0) * tf.square(loss_penalty) + alpha_tf * loss_penalty
            loss_obj = loss_obj + lambda_tf * tf.reduce_sum(tf.abs(W_tf))

            optim = tf.compat.v1.train.AdamOptimizer(d_lrd, beta1=0.9)
            train_op = optim.minimize(loss_obj)

            reset_optimizer_op = tf.compat.v1.variables_initializer(optim.variables())
            model_init = tf.compat.v1.global_variables_initializer()

        self.graph = graph
        self.graph_nodes = {
            "sem_data": X_tf,
            "weight": W_tf,
            "rho": rho_tf,
            "lambda": lambda_tf,
            "opt_step": d_lrd,
            "alpha": alpha_tf,
            "loss_penalty": loss_penalty,
            "loss_regress": loss_xw,
            "loss_obj": loss_obj,
            "train_op": train_op,
            "reset_opt": reset_optimizer_op,
            "init_vars": model_init,
        }

        return self.graph_nodes

    def model_train(
        self,
        sess: tf.compat.v1.Session,
        outer_iter: int = 50,
        inner_iter: int = 100,
        init_global_step: float = 1e-2,
        h_tol: float = 1e-8,
        verbose: bool = False,
    ):
        """
        Runs the augmented Lagrangian optimization to fit the NOTEARS weights.
        """
        h = np.inf
        for _ in range(outer_iter):
            sess.run(self.graph_nodes["reset_opt"])

            for t1 in range(inner_iter):
                feed_dict = {
                    self.graph_nodes["alpha"]: self.alpha,
                    self.graph_nodes["rho"]: self.rho,
                    self.graph_nodes["opt_step"]: init_global_step / np.sqrt(1.0 + t1),
                    self.graph_nodes["lambda"]: self.lamb,
                }
                sess.run(self.graph_nodes["train_op"], feed_dict=feed_dict)

            h = sess.run(self.graph_nodes["loss_penalty"], feed_dict=feed_dict)

            if verbose:
                loss_val = sess.run(self.graph_nodes["loss_obj"], feed_dict=feed_dict)
                print(f"[NOTEARS] penalty={h:.3e}, loss={loss_val:.3e}, rho={self.rho:.3e}")

            if h <= h_tol:
                break

            self.alpha += self.rho * h
            self.rho *= self.rho_multiplier


def estimate_notears_weights(
    X: np.ndarray,
    lambda1: float = 0.05,
    alpha_init: float = 0.0,
    rho_init: float = 1.0,
    rho_multiplier: float = 1.25,
    outer_iter: int = 50,
    inner_iter: int = 100,
    step_size: float = 1e-2,
    h_tol: float = 1e-8,
    verbose: bool = False,
) -> np.ndarray:
    """
    Trains NOTEARS and returns the learned weighted adjacency matrix.
    """
    trainer = NoTearTF(
        lamb=lambda1,
        alpha_init=alpha_init,
        rho_init=rho_init,
        rho_multiplier=rho_multiplier,
    )
    graph_nodes = trainer.construct_graph(X)

    with tf.compat.v1.Session(graph=trainer.graph) as sess:
        sess.run(graph_nodes["init_vars"])
        trainer.model_train(
            sess,
            outer_iter=outer_iter,
            inner_iter=inner_iter,
            init_global_step=step_size,
            h_tol=h_tol,
            verbose=verbose,
        )
        weights = sess.run(graph_nodes["weight"])

    weights = np.asarray(weights, dtype=np.float64)
    np.fill_diagonal(weights, 0.0)
    return weights

