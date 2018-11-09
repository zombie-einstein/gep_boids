import numpy as np
import random as rnd
import operator
import dask


class Boid:
    def __init__(self, w, h, max_vel, min_vel, max_rot, max_vis):
        self.spc = np.array([w, h])
        self.max_vel = max_vel
        self.min_vel = min_vel
        self.max_rot = max_rot
        self.max_vis = max_vis
        self.v = rnd.uniform(min_vel, max_vel)
        self.t = rnd.uniform(0, 1.0)
        self.pos = np.multiply(np.random.rand(2,), self.spc)
    
    def move(self):
        """Update position based on current velocity and heading, taking care of wrapping on torus"""
        self.pos = np.mod(self.pos + self.v*np.array([np.cos(self.t*2*np.pi), np.sin(self.t*2*np.pi)]), self.spc)
    
    def rotate(self, t):
        """Rotate heading, up to a maximum angle of rotation"""
        self.t = (self.t + max(min(self.max_rot, t), -self.max_rot)) % 1.0
        
    def accelerate(self, v):
        """Accelerate/Decelerate inside a min/max value"""
        self.v = max(min(self.max_vel, self.v+v), self.min_vel)

    def dist_to(self, x):
        return np.linalg.norm(x-self.pos)
    
    def angle_to(self, x):
        d = x - self.pos
        return (np.arctan2(d[1], d[0]) % (2*np.pi))/(2*np.pi)
    
    def get_telemetry(self):
        return self.pos, self.v, self.t
    
    def get_neighbours(self, b_list, n=3):
        nbs = [(self.dist_to(x[0]), self.angle_to(x[0]), x[1], x[2]) for x in b_list]
        return np.array(sorted([x for x in nbs if x[0] < self.max_vis], key=operator.itemgetter(0))[:n])

    def step(self):
        self.rotate(rnd.uniform(-1.0, 1.0))
        self.accelerate(rnd.uniform(-0.5, 0.5))
        self.move()


class Flock:
    def __init__(self, w, h, n,  max_vel, min_vel, max_rot, max_vis):
        self.boids = [Boid(w, h, max_vel, min_vel, max_rot, max_vis) for _ in range(n)]
        self.boid_steps = dask.delayed([dask.delayed(x.step())() for x in self.boids])
        
    def step(self):
        dask.delayed([dask.delayed(x.step())() for x in self.boids]).compute()
        #self.boid_steps.compute()
            
    def run(self, steps):
        history = []
        for _ in range(steps):
            history.append([x.pos for x in self.boids])
            self.step()
        return np.stack([np.vstack(x) for x in history])
        
    def run_and_animate(self, steps):
        history = self.run(steps)
        import matplotlib.pyplot as plt
        from matplotlib import animation

        fig = plt.figure()
        ax = plt.axes(xlim=(0, 10.0), ylim=(0, 10.0))
        time_text = ax.text(0.05, 0.95, '', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        line, = ax.plot([], [], 'o')

        def animate(i):
            line.set_data(history[i, :, 0], history[i, :, 1])  # update the data
            time_text.set_text('t {:3}'.format(i))
            return [line] + [time_text]

        return animation.FuncAnimation(fig, animate, frames=steps, interval=80, blit=True)