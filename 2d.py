from FEM import *

if __name__ == "__main__":
    def exact(x):
        return np.where(x < 1/2, 2*x, 2*(1 - x))

    def f(x):
        return np.where(x < 1/2, 2, -2)


    u = solve_system(f, exact=exact, N=100)

    u.plot_comparison()
    plt.legend()
    plt.xlabel('x')

    plt.show()

    ax = plt.subplot()

    u.plot_error(ax=ax)
    ax.legend()
    ax.set_xlabel('x')

    plt.show()

    plot_convergence(f, exact, title="Convergence of solver")