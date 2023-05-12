import numpy as np
import pycpp_examples
from pytest import approx
import time


def test_pycpp_pose():
    print(pycpp_examples)
    print(dir(pycpp_examples))
    Pose = pycpp_examples.Pose
    pa = Pose(0, 0.2, 0.1)
    pb = Pose(x=0.0, y=0.2, yaw=0.2)

    odom = Pose.get_odom(pa, pb)
    assert odom.x == approx(0.0)
    assert odom.y == approx(0.0)
    assert odom.yaw == approx(-0.1)


def test_pycpp_vector_ops():
    """Show how PyBind works with vectors/lists.
    Generates random data and sqrt-sum's in different ways."""

    # Generate some random data
    vec = np.random.random(500000)

    start = time.time()
    sum_np = np.sqrt(vec).sum()
    print(f"Numpy Sum Time: {time.time() - start}")

    start = time.time()
    sum_cpp_eigen = pycpp_examples.sqrt_sum_vec(vec)
    print(f"C++ Eigen Sum Time: {time.time() - start}")

    # Convert to a vector
    vec = list(vec)

    start = time.time()
    sum_loop = 0
    for v in vec:
        sum_loop += np.sqrt(v)
    print(f"Python loop sum time: {time.time() - start}")

    start = time.time()
    sum_cpp_vec = pycpp_examples.sqrt_sum_vec(vec)
    print(f"C++ vector sum time: {time.time() - start}")

    assert sum_np == approx(sum_cpp_eigen)
    assert sum_np == approx(sum_loop)
    assert sum_np == approx(sum_cpp_vec)
