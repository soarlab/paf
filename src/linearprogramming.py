import numpy as np
from scipy.optimize import linprog

class LP_Instance:
    def __init__(self, marginal_left, marginal_right, insides):
        self.i=0
        self.association={}
        self.constraints=[]
        self.values=[]
        self.marginal_left=marginal_left
        self.marginal_right=marginal_right
        self.internals=insides
        self.evaluation_points=set()
        self.elaborate_constraints()
        self.compute_evaluation_points()

    def elaborate_constraints(self):
        for pbox in self.marginal_left:
            for kid in pbox.kids:
                if kid not in self.association:
                    self.association[kid]=self.i
                    self.i=self.i+1
        for pbox in self.marginal_right:
            for kid in pbox.kids:
                if kid not in self.association:
                    self.association[kid] = self.i
                    self.i = self.i + 1
        for pbox in self.marginal_left:
            vect=np.zeros(len(self.association))
            for kid in pbox.kids:
                vect[self.association[kid]]=1
            self.constraints.append(vect)
            self.values.append(pbox.prob)
        for pbox in self.marginal_right:
            vect=np.zeros(len(self.association))
            for kid in pbox.kids:
                vect[self.association[kid]]=1
            self.constraints.append(vect)
            self.values.append(pbox.prob)

    def compute_evaluation_points(self):
        for pbox in self.internals:
            self.evaluation_points.add(pbox.lower)
            self.evaluation_points.add(pbox.upper)
        #for pbox in self.marginal_right:
        #    self.evaluation_points.add(pbox.lower)
        #    self.evaluation_points.add(pbox.upper)
        self.evaluation_points=sorted(self.evaluation_points)

    def optimize_max(self):
        edge_cdf=[]
        val_cdf = []
        for ev_point in self.evaluation_points:
            vect = np.zeros(len(self.association))
            for intern in self.internals:
                if intern.lower <= ev_point:
                    vect[self.association[intern]]=1.0
            tmp_constraints=[]
            tmp_b=[]
            unique_constraints, index_constraints = np.unique(self.constraints, axis=0, return_index=True)
            for ind, val in enumerate(unique_constraints):
                tmp_constraints.append(val)
                tmp_b.append(self.values[index_constraints[ind]])

            print("Evaluation Point", ev_point)
            res = linprog(-vect, A_eq=np.array(tmp_constraints), b_eq=np.array(tmp_b), bounds=(0, None))
            edge_cdf.append(ev_point)
            val_cdf.append(-res.fun)
        return edge_cdf, val_cdf

    def optimize_min(self):
        edge_cdf = []
        val_cdf = []
        for ev_point in self.evaluation_points:
            vect = np.zeros(len(self.association))
            for intern in self.internals:
                if intern.upper <= ev_point:
                    vect[self.association[intern]]=1.0
            tmp_constraints=[]
            tmp_b=[]
            unique_constraints, index_constraints = np.unique(self.constraints, axis=0, return_index=True)
            for ind, val in enumerate(unique_constraints):
                tmp_constraints.append(val)
                tmp_b.append(self.values[index_constraints[ind]])

            print("Evaluation Point", ev_point)
            res = linprog(vect, A_eq=np.array(tmp_constraints), b_eq=np.array(tmp_b), bounds=(0, None))
            edge_cdf.append(ev_point)
            val_cdf.append(res.fun)
        return edge_cdf, val_cdf