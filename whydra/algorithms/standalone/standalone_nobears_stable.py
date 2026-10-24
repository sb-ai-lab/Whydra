import numpy as np
from typing import Dict, Any, Optional

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - informative error path
    raise ImportError(
        "TensorFlow is required for the NOBEARS implementation. "
        "Please install the dependencies listed in requirements.txt."
    ) from exc

from scipy.optimize import lsq_linear
from scipy.stats import spearmanr


# NOBEARS uses TF1-style graphs (placeholders, sessions).
# Keep compatibility with TF2 by enabling the v1 API explicitly.
tf.compat.v1.disable_eager_execution()


def W_reg_init(X: np.ndarray) -> np.ndarray:
    """
    Heuristic initialization for the weight matrix W based on pairwise
    rank correlations
    """
    X = np.asarray(X, dtype=np.float64)  # ensure floating point
    n, d = X.shape
    rho, _ = spearmanr(X)
    rho = np.asarray(rho, dtype=float)
    np.fill_diagonal(rho, 0.0)

    W_init = np.zeros((d, d), dtype=float)
    for i in range(d):
        ind = np.argsort(rho[:, i])[::-1][: min(int(n * 0.5), 30)]
        Xpar = np.asarray(X[:, ind], dtype=np.float64).copy()
        Xpar += 0.01 * np.random.randn(*Xpar.shape)
        yc = X[:, i].astype(np.float64, copy=False)
        res = lsq_linear(Xpar, yc)
        wpar = res["x"]
        W_init[ind, i] = wpar

    return W_init


def power_iteration_tf(W, power_iteration_rounds: int = 1):
    """
    Power iteration to approximate the largest singular value of W.

    Args:
        W: 2D TF tensor (d x d)
        power_iteration_rounds: number of power iteration steps
    """
    d = int(W.shape[0])

    # Right eigenvector
    u_var = tf.compat.v1.get_variable(
        W.name.replace(":", "") + "/u_var",
        shape=(d, 1),
        dtype=W.dtype,
        initializer=tf.compat.v1.random_uniform_initializer(),
        trainable=False,
    )

    # Left eigenvector
    v_var = tf.compat.v1.get_variable(
        W.name.replace(":", "") + "/v_var",
        shape=(d, 1),
        dtype=W.dtype,
        initializer=tf.compat.v1.random_uniform_initializer(),
        trainable=False,
    )

    u = u_var
    v = v_var

    for _ in range(power_iteration_rounds):
        v = tf.nn.l2_normalize(tf.matmul(W, v, transpose_a=True), axis=None, epsilon=1e-12)
        u = tf.nn.l2_normalize(tf.matmul(W, u), axis=None, epsilon=1e-12)

    with tf.control_dependencies([tf.compat.v1.assign(u_var, u, name="update_u")]):
        u = tf.identity(u)

    with tf.control_dependencies([tf.compat.v1.assign(v_var, v, name="update_v")]):
        v = tf.identity(v)

    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)

    # Largest singular value of W.
    norm_value = tf.matmul(tf.matmul(v, W, transpose_a=True), u) / tf.reduce_sum(v * u)
    norm_value.shape.assert_is_fully_defined()
    norm_value.shape.assert_is_compatible_with([1, 1])

    return u_var, v_var, u, v, norm_value


class NoBearsTF:
    """
    TensorFlow implementation of the NOBEARS optimization routine
    """

    def __init__(
        self,
        beta1: float = 0.05,
        beta2: float = 0.001,
        alpha_init: float = 0.01,
        rho_init: float = 1.0,
        poly_degree: int = 3,
        l1_ratio: float = 0.5,
    ):
        self.alpha = float(alpha_init)
        self.rho = float(rho_init)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.poly_degree = int(poly_degree)
        self.l1_ratio = float(l1_ratio)
        self.graph_nodes: Dict[str, Any] = {}
        self.graph: Optional[tf.Graph] = None

    # ------------------------------------------------------------------ #
    # Graph construction                                                 #
    # ------------------------------------------------------------------ #
    def construct_graph(
        self,
        X: np.ndarray,
        W_init: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Builds the TF computation graph for the NOBEARS objective.

        Args:
            X: Data matrix of shape (n_samples, n_features).
            W_init: Optional initialization matrix for W.
        """
        n, d = X.shape
        graph = tf.Graph()

        with graph.as_default():
            X_tf = tf.constant(X.astype("float32"), name="sem_data")  # n x d
            batch_size_tf = tf.cast(tf.shape(X_tf)[0], tf.float32)

            if W_init is None:
                W_var = tf.compat.v1.get_variable(
                    "W",
                    shape=(d, d),
                    initializer=tf.compat.v1.random_uniform_initializer(),
                )
            else:
                W_init_tf = W_init.astype("float32")
                W_var = tf.compat.v1.get_variable(
                    "W",
                    initializer=W_init_tf,
                    dtype=tf.float32,
                )

            W_tf = tf.linalg.set_diag(W_var, tf.zeros(d, dtype=tf.float32))

            # Exponential moving average of W
            ema = tf.train.ExponentialMovingAverage(decay=0.975)
            ema_op = ema.apply([W_tf])
            tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, ema_op)
            W_ema = ema.average(W_tf)

            noise_level_tf = tf.compat.v1.placeholder(tf.float32, name="noise_level")
            X0_tf = X_tf + noise_level_tf * tf.random.normal(tf.shape(X_tf), stddev=1.0)

            # Polynomial regression part
            bk0_tf = tf.compat.v1.get_variable(
                "bk_poly0",
                shape=(1, d),
                initializer=tf.compat.v1.random_uniform_initializer(),
            )
            X1_tf = bk0_tf

            regression_vars = [bk0_tf]
            for k in range(1, self.poly_degree + 1):
                sig_tf = tf.compat.v1.get_variable(
                    f"sig_poly{k}",
                    shape=(1, d),
                    initializer=tf.compat.v1.random_uniform_initializer(),
                )
                regression_vars.append(sig_tf)
                if k == 1:
                    X1_tf = X1_tf + X0_tf * sig_tf
                else:
                    X1_tf = X1_tf + (X0_tf ** k) * sig_tf

            X2_tf = tf.matmul(X1_tf, W_tf)

            loss_xw = tf.reduce_sum((X2_tf - X_tf) ** 2) / (2.0 * batch_size_tf)

            W2 = W_tf * W_tf
            W2p = W2 + 1e-6

            _, _, _, _, h_tf = power_iteration_tf(W2p, power_iteration_rounds=5)
            loss_penalty = tf.reduce_sum(h_tf)

            rho_tf = tf.compat.v1.placeholder(tf.float32, name="rho")
            beta1_tf = tf.compat.v1.placeholder(tf.float32, name="beta1")
            beta2_tf = tf.compat.v1.placeholder(tf.float32, name="beta2")
            d_lrd = tf.compat.v1.placeholder(tf.float32, name="opt_step")
            alpha_tf = tf.compat.v1.placeholder(tf.float32, name="alpha")

            # Elastic-net style regularization on W
            W_abs = tf.abs(W_tf)
            reg0 = tf.reduce_sum(W_abs)
            reg1 = tf.reduce_sum(W2) * 0.5

            l1_ratio = self.l1_ratio
            if l1_ratio < 1e-5:
                loss_reg = beta1_tf * reg1
            elif l1_ratio > 0.99999:
                loss_reg = beta1_tf * reg0
            else:
                loss_reg = beta1_tf * ((1.0 - l1_ratio) * reg1 + l1_ratio * reg0)

            for w in regression_vars:
                loss_reg = loss_reg + beta2_tf * tf.reduce_sum(w ** 2)

            loss_init = loss_xw + loss_reg
            loss_obj = loss_init + (rho_tf / 2.0) * (loss_penalty ** 2) + alpha_tf * loss_penalty

            # Optimizers
            optim = tf.compat.v1.train.AdamOptimizer(d_lrd, beta1=0.9)
            sn_ops = [
                op
                for op in tf.compat.v1.get_default_graph().get_operations()
                if "update_u" in op.name or "update_v" in op.name
            ]

            with tf.control_dependencies(sn_ops):
                train_op = optim.minimize(loss_obj)

            optim_init = tf.compat.v1.train.AdamOptimizer(d_lrd, beta1=0.9)
            with tf.control_dependencies(sn_ops):
                init_train_op = optim_init.minimize(loss_init)

            reset_optimizer_op = tf.compat.v1.variables_initializer(optim.variables())
            model_init = tf.compat.v1.global_variables_initializer()

        self.graph = graph
        self.graph_nodes = {
            "sem_data": X_tf,
            "weight": W_tf,
            "weight_ema": W_ema,
            "rho": rho_tf,
            "loss_penalty": loss_penalty,
            "beta1": beta1_tf,
            "beta2": beta2_tf,
            "opt_step": d_lrd,
            "alpha": alpha_tf,
            "noise_level": noise_level_tf,
            "loss_regress": loss_xw,
            "loss_obj": loss_obj,
            "train_op": train_op,
            "reset_opt": reset_optimizer_op,
            "update_uv": sn_ops,
            "init_vars": model_init,
            "moving_average_op": ema_op,
            "init_train_op": init_train_op,
        }

        return self.graph_nodes

    # ------------------------------------------------------------------ #
    # Training loops                                                     #
    # ------------------------------------------------------------------ #
    def model_init_train(
        self,
        sess: tf.compat.v1.Session,
        init_iter: int = 200,
        init_global_step: float = 1e-2,
        noise_level: float = 0.1,
    ) -> None:
        """
        Initial warm-up training phase (without augmented Lagrangian penalty)
        """
        for t1 in range(init_iter):
            feed_dict = {
                self.graph_nodes["alpha"]: self.alpha,
                self.graph_nodes["rho"]: self.rho,
                self.graph_nodes["opt_step"]: init_global_step / np.sqrt(1.0 + 0.1 * t1),
                self.graph_nodes["beta1"]: self.beta1,
                self.graph_nodes["beta2"]: self.beta2,
                self.graph_nodes["noise_level"]: noise_level,
            }
            sess.run(self.graph_nodes["init_train_op"], feed_dict=feed_dict)

    def model_train(
        self,
        sess: tf.compat.v1.Session,
        outer_iter: int = 200,
        inner_iter: int = 100,
        init_global_step: float = 1e-2,
        noise_level: float = 0.1,
    ) -> None:
        """
        Main augmented Lagrangian optimization loop.
        """
        ave_loss_reg = -1.0
        ave_loss_min = np.inf

        for _ in range(outer_iter):
            sess.run(self.graph_nodes["reset_opt"])
            v0 = 0.0
            h_val = 0.0

            for t1 in range(inner_iter):
                feed_dict = {
                    self.graph_nodes["alpha"]: self.alpha,
                    self.graph_nodes["rho"]: self.rho,
                    self.graph_nodes["opt_step"]: init_global_step / np.sqrt(1.0 + t1),
                    self.graph_nodes["beta1"]: self.beta1,
                    self.graph_nodes["beta2"]: self.beta2,
                    self.graph_nodes["noise_level"]: noise_level,
                }

                _, v0, h_val = sess.run(
                    [
                        self.graph_nodes["train_op"],
                        self.graph_nodes["loss_regress"],
                        self.graph_nodes["loss_penalty"],
                    ],
                    feed_dict=feed_dict,
                )

            sess.run(self.graph_nodes["moving_average_op"])

            if ave_loss_reg < 0:
                ave_loss_reg = v0
            else:
                ave_loss_reg = 0.1 * ave_loss_reg + 0.9 * v0

            if ave_loss_min > ave_loss_reg:
                ave_loss_min = ave_loss_reg
            elif ave_loss_reg > (1.5 * ave_loss_min) and h_val < 0.005:
                # Early stopping condition from the reference implementation
                return

            self.alpha += self.rho * h_val
            self.rho = min(1e8, 1.1 * self.rho)


def estimate_nobears_weights(
    X: np.ndarray,
    beta1: float = 0.05,
    beta2: float = 0.001,
    alpha_init: float = 0.01,
    rho_init: float = 1.0,
    poly_degree: int = 3,
    l1_ratio: float = 0.5,
    init_iter: int = 200,
    outer_iter: int = 200,
    inner_iter: int = 100,
    step_size: float = 1e-2,
    noise_level: float = 0.1,
    use_reg_init: bool = True,
) -> np.ndarray:
    """
    High-level helper that trains NOBEARS and returns the learned
    weighted adjacency matrix.
    """
    W_init = W_reg_init(X) if use_reg_init else None

    trainer = NoBearsTF(
        beta1=beta1,
        beta2=beta2,
        alpha_init=alpha_init,
        rho_init=rho_init,
        poly_degree=poly_degree,
        l1_ratio=l1_ratio,
    )
    graph_nodes = trainer.construct_graph(X, W_init=W_init)

    with tf.compat.v1.Session(graph=trainer.graph) as sess:
        sess.run(graph_nodes["init_vars"])
        trainer.model_init_train(
            sess,
            init_iter=init_iter,
            init_global_step=step_size,
            noise_level=noise_level,
        )
        trainer.model_train(
            sess,
            outer_iter=outer_iter,
            inner_iter=inner_iter,
            init_global_step=step_size,
            noise_level=noise_level,
        )
        weights = sess.run(graph_nodes["weight_ema"])

    weights = np.asarray(weights, dtype=np.float64)
    np.fill_diagonal(weights, 0.0)
    return weights


