from collections import deque
from random import choice, random
from random import randint, uniform
import numpy as np


def kwarg_return(x):
    def foo(**kwargs):
        return kwargs[x]
    return foo


def list_return(x):
    def foo(arr):
        return arr[x]
    return foo


def get_func_node(f, args):
    def foo(arr):
        return f(*[i(arr) for i in args])
    return foo


class LeafNode:
    """Leaf node, just returns value from an argument array, obviously has no children nodes"""
    def __init__(self, x):
        self.x = x
        self.n = 0
    
    def process(self):
        return list_return(self.x)


class FuncNode:
    """Intermediate function leaf nodes, return their function with child return values passed as list of arguments"""
    def __init__(self, func, n):
        self.func = func
        self.n = n
        self.child = []
    
    def process(self):
        """Return the value of this node from results of children nodes. A func node may be the
        root of the function tree and so this function returns the final result of the tree function"""
        return get_func_node(self.func, [i.process() for i in self.child])


def crossover_1(ga, gb):
    """One-point random crossover"""
    cp = randint(0, len(ga))
    return ga[:cp] + gb[cp:], gb[:cp] + ga[cp:]


def crossover_2(ga, gb):
    """2-Point random crossover"""
    cps = [randint(0, len(ga)), randint(0, len(ga))]
    cps.sort()
    return ga[:cps[0]] + gb[cps[0]:cps[1]] + ga[cps[1]:], gb[:cps[0]] + ga[cps[0]:cps[1]] + gb[cps[1]:]


class GEP:
    """GEP utility class initialized on a allele <-> function mapping dictionary and input size
    (which will be passed as list to the generated function). This class both generates random genotypes based on the
    function/input mappings and converts said genotypes into executable function graphs"""

    def __init__(self, func_map, n_args, len_h):
        self.func_map = func_map
        self.n_args = n_args
        self.len_h = len_h
        self.max_args = max([i['n'] for i in self.func_map.values()])
        self.len_t = len_h * (self.max_args - 1) + 1
        self.len_g = self.len_h + self.len_t
        self.index = [str(i) for i in range(self.n_args)]
        self.head_alleles = list(self.func_map.keys())+self.index

    def make_node(self, c):
        """Return an appropriate node based on an input character"""
        if c.isdigit():
            return LeafNode(int(c))
        else:
            f = self.func_map[c]
            return FuncNode(f['func'], f['n'])
    
    def pre_phenotype(self, genome):
        """From a genotype string return a executable function composition and the number of nodes used.
        This uses prefix-gene expression, so we visit nodes in DFS pre-order"""
        root = self.make_node(genome[0])
        a = [root]
        b = deque(genome[1:])
        count = 1
        while a:
            if a[-1].n > 0:
                a[-1].n -= 1
                count += 1
                new_node = self.make_node(b.popleft())
                a[-1].child.append(new_node)
                a.append(new_node)
            else:
                a.pop()

        return root.process(), count
    
    def phenotype(self, genome):
        """From a genotype string return a executable function composition and the number of nodes used"""
        root = self.make_node(genome[0])
        a = deque([root])
        b = deque(genome[1:])
        count = 1
        while a:
            count += 1
            curr = a.popleft()
            for i in range(curr.n):
                new_node = self.make_node(b.popleft())
                curr.child.append(new_node)
                a.append(new_node)
    
        return root.process(), count

    def random_genome(self):
        """Generate a random genome fitting the class parameters with argument genome head length"""
        ret = ''
        for i in range(self.len_h):
            ret += choice(self.head_alleles)
        for i in range(self.len_t):
            ret += choice(self.index)
        return ret
    
    def mutate_genome(self, mutation_rate, genome):
        """Randomly mutate alleles, acting on head and tail alleles accordingly"""
        ret = ""
        for i in range(self.len_h):
            ret += genome[i] if random() > mutation_rate else choice(self.head_alleles)
        for i in range(self.len_h, self.len_h+self.len_t):
            ret += genome[i] if random() > mutation_rate else choice(self.index)
        return ret

    def breed(self, population, fitness, mutation_rate=0.01):
        """For argument population of genes, and corresponding fitness values crossover genes
        selecting for fitness, as well as randomly mutate. Returns list of new gene population"""
        new_pop = []
        cum_sum = np.cumsum(fitness - np.min(fitness))
        for j in range(int(len(population) / 2)):
        
            n1 = uniform(0.0, cum_sum[-1])
            i1 = 0
            while n1 > cum_sum[i1]:
                i1 += 1
        
            n2 = uniform(0.0, cum_sum[-1])
            i2 = 0
            while n2 > cum_sum[i2]:
                i2 += 1
        
            new1, new2 = crossover_1(population[i1], population[i2])
            new1 = self.mutate_genome(mutation_rate, new1)
            new2 = self.mutate_genome(mutation_rate, new2)
            new_pop.extend([new1, new2])
    
        return new_pop


def tree_test():
    """Tree building sanity check. Just prints BFS structure of a example tree"""
    A = GEP({'s': {'func': lambda x, y: print('sum'), 'n': 2}, 'm': {'func': lambda x, y: print('Minus'), 'n': 2}},
            2, 5)
    R = A.pre_phenotype('sms011m01')
    D = deque([R])
    while D:
        curr = D.popleft()
        if isinstance(curr, FuncNode):
            curr.func(1, 2)
            for i in curr.child:
                D.append(i)
        else:
            print(curr.x)
