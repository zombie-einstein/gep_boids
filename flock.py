import numpy as np
import numpy.ma as ma
from sklearn.metrics.pairwise import pairwise_distances as pwd


class FlockState:
    def __init__(self, flock_size, space_dims, max_vel, min_vel, max_rot, max_vis):
        """
        flock_size: Number of agents
        space_dims: List of spatial bounds (i.e. max value) for each dimension
        max_vel: Max length of velocity vector (i.e. in polar co-ords)"""
        assert len(space_dims) == 2
        self.flock_size = flock_size
        self.spc_dim = space_dims
        self.max_vel = max_vel
        self.min_vel = min_vel
        self.max_rot = max_rot
        self.max_vis = max_vis
        self.spc = np.multiply(np.random.random_sample((flock_size, 2)), np.array(space_dims, np.newaxis))
        self.vel = np.concatenate([np.multiply(np.random.random_sample((flock_size, 1))*2-1,
                                               np.array([np.pi], np.newaxis)),
                                   np.multiply(np.random.random_sample((flock_size, 1)),
                                               np.array(max_vel, np.newaxis))], axis=1)
        self.mask = np.zeros((flock_size, flock_size), dtype=bool)
        np.fill_diagonal(self.mask, True)
        self.rel_dist = self.masked_distances()
        self.fill_distances()
        self.cum_rewards = np.array([0.0 for _ in range(flock_size)])
        
    def masked_distances(self):
        return ma.masked_array(pwd(self.spc), self.mask)
        
    def fill_distances(self):
        """Fill distance matrix for all pairs of boids generating a n x n distance matrix
        set diagonals to infinity as we should effectively not be using ourselves in measurements"""
        self.rel_dist = self.masked_distances()
            
    def update(self):
        """Update function, updates all positions according to their current """
        cartesian = self.vel[:, 1:2]*np.concatenate([np.cos(self.vel[:, 0:1]), np.sin(self.vel[:, 0:1])], axis=1)
        self.spc = np.concatenate([(self.spc[:, 0:1] + cartesian[:, 0:1]) % self.spc_dim[0],
                                   (self.spc[:, 1:2] + cartesian[:, 1:2]) % self.spc_dim[1]], axis=1)
        
    def printer(self):
        print('{:8} {:8} {:8} {:8}'.format('x', 'y', 'vr', 'theta'))
        for i in range(self.flock_size):
            print('{:8.3f} {:8.3f} {:8.3f} {:8.3f}'.format(*self.spc[i], *self.vel[i]))
        
    def update_angle(self, d_vel):
        """Update velocities from argument array, restricting rotation to a max turning rate"""
        self.vel[:, 0:1] += (np.clip(d_vel, -self.max_rot, self.max_rot)).reshape(-1, 1)
        self.vel[:, 0:1] = (self.vel[:, 0:1] + np.pi) % (2 * np.pi) - np.pi
        
    def update_vel(self, d_vel):
        """Update velocities, restricting them to range inside min/max velocities"""
        self.vel[:, 1:2] = np.clip(self.vel[:, 1:2] + d_vel.reshape(-1, 1), self.min_vel, self.max_vel)
        
    def closest_neighbours(self, n=3):
        """For each boid get a list of indices of nearest neighbours that fall inside a visibility range"""
        for x in self.rel_dist:
            idx, = ma.where(x < self.max_vis)
            yield idx[ma.argsort(x[idx])][:n]
    
    def local_state(self, n=3):
        """Get details of nearest neighbours for each boid"""
        for i, x in enumerate(self.closest_neighbours(n)):
            ret = []
            for y in x:
                ret.extend(self.vel[y])
                ret.extend(self.spc[y])
            ret += [-1.0] * (4*n - len(ret))
            ret.extend(self.vel[i])
            ret.extend(self.spc[i])
            yield ret
    
    def reset(self):
        """Reset all boids to random positions and velocities, and reset distances and rewards"""
        self.spc = np.multiply(np.random.random_sample((self.flock_size, 2)), np.array(self.spc_dim, np.newaxis))
        self.vel = np.concatenate([np.multiply(np.random.random_sample((self.flock_size, 1)) * 2 - 1,
                                               np.array([np.pi], np.newaxis)),
                                   np.multiply(np.random.random_sample((self.flock_size, 1)),
                                               np.array(self.max_vel, np.newaxis))], axis=1)
        self.rel_dist = self.masked_distances()
        self.fill_distances()
        self.cum_rewards = np.array([0.0 for _ in range(self.flock_size)])
    
    def rewards(self):
        self.cum_rewards += np.sum(5-0.75*self.rel_dist, axis=1)\
                            - (self.rel_dist < 0.005).sum(axis=1)*500\
                            + (1 - self.max_vel/self.vel[:, 1:2]).sum(axis=1)*100
