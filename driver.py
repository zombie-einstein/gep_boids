import numpy as np
from flock import FlockState
from genome import GEP


f_map = {'S': {'func': lambda x, y: x+y, 'n': 2},
         'M': {'func': lambda x, y: x-y, 'n': 2},
         'T': {'func': lambda x, y: x*y, 'n': 2},
         'D': {'func': lambda x, y: x/y if y > 0 else 0.0, 'n': 2},
         'C': {'func': lambda x: np.cos(x), 'n': 1},
         'I': {'func': lambda x: np.sin(x), 'n': 1},
         'N': {'func': lambda x: -x, 'n': 1},
         'A': {'func': lambda x: x+1.0, 'n': 1},
         'R': {'func': lambda x: x-1.0, 'n': 1}}


class Flock:
    def __init__(self, flock_size, l_gene, space_dim, max_vel, min_vel):
        assert flock_size % 2 == 0
        self.flock_size = flock_size
        self.state = FlockState(flock_size, space_dim, max_vel, min_vel, np.pi/16, 30.0)
        self.gep = GEP(f_map, 4*4, l_gene)
        self.dv_genes = [self.gep.random_genome() for _ in range(flock_size)]
        self.dt_genes = [self.gep.random_genome() for _ in range(flock_size)]
        self.dv_pheno = [self.gep.pre_phenotype(i)[0] for i in self.dv_genes]
        self.dt_pheno = [self.gep.pre_phenotype(i)[0] for i in self.dt_genes]
        self.steps = []
        
    def update_step(self):
        self.steps.append(self.state.spc)
        self.state.fill_distances()
        dv = []
        dt = []
        for i, j in enumerate(self.state.local_state()):
            dv.append(self.dv_pheno[i](j))
            dt.append(self.dt_pheno[i](j))
        self.state.update_vel(np.array(dv))
        self.state.update_angle(np.array(dt))
        self.state.update()
        self.state.rewards()
    
    def plot_trajectories(self):
        import matplotlib.pyplot as plt
        for i in range(self.flock_size):
            plt.plot(*np.stack(self.steps)[:, i, :].T, label='B {}'.format(i))
        plt.legend()
        plt.show()

    def animated_plot(self, path):
        import matplotlib.pyplot as plt
        from matplotlib import animation
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.state.spc_dim[0]), ylim=(0, self.state.spc_dim[1]))
        time_text = ax.text(0.05, 0.95, '', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        line, = ax.plot([], [], 'o')
        
        def animate(i):
            line.set_data(self.steps[i][:, 0], self.steps[i][:, 1])  # update the data
            time_text.set_text('t {:3}'.format(i))
            return [line] + [time_text]

        return animation.FuncAnimation(fig, animate, frames=len(self.steps), interval=80, blit=True)

    def breed(self, fitness):
        self.dv_genes = self.gep.breed(self.dv_genes, fitness)
        self.dt_genes = self.gep.breed(self.dt_genes, fitness)
        self.dv_pheno = [self.gep.pre_phenotype(i)[0] for i in self.dv_genes]
        self.dt_pheno = [self.gep.pre_phenotype(i)[0] for i in self.dt_genes]

    def full_gen(self, n):
        self.steps = []
        fitness = np.array([0.0 for _ in range(self.flock_size)])
        for _ in range(5):
            self.state.reset()
            for _ in range(n):
                self.update_step()
            fitness += self.state.cum_rewards
        self.breed(fitness)


A = Flock(20, 20, [500.0, 500.0], 4.0, 0.1)
np.set_printoptions(threshold=np.nan)
np.core.arrayprint._line_width = 400

for a in range(200):
    A.full_gen(800)
    scrs = A.state.cum_rewards
    print("Gen {:4}, Avg: {:12.2f}, Min{:12.2f}, Max: {:12.2f}".format(a,
                                                                     np.mean(scrs),
                                                                     np.min(scrs),
                                                                     np.max(scrs)))

print(A.dt_genes)
print(A.dv_genes)
print(A.state.vel)
A.animated_plot(None)
