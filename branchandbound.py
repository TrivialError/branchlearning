import gurobipy as grb
import networkx as nx
import queue
import math
import drawsolution

# TODO: check if it's actually working

# lp_initializer should generate a gurobi model and variable objects in a tupledict given a graph
# branch_rule should take in the graph and an assignment to the variables and a gurobi model
#   and return a variable to branch on
# items in queue have format (LPvalue, ({vars tupledict}, [branch_values]))
# init_soln is of the form (solution_obj_value, {vars tupledict})
# cutting_planes should take the graph and lp solution and return constraints as [(LinExpr, rhs)]
# TODO: simplify stuff a bit by not passing actual variable objects around; just use a tupledict with the
#   edge index as keys and the name as values; then get the variable from the relevant model as needed


class BranchAndBound:

    def __init__(self, branch_rule, lp_initializer, graph, init_soln=(math.inf, ([], []))):
        self.branch_rule = branch_rule
        self.lp, self.x = lp_initializer(graph)
        self.lp.params.outputflag = 0
        self.graph = graph
        self.queue = queue.PriorityQueue()

        self.best_soln = init_soln

    def solve(self, draw=False):
        self.queue.put((math.inf, ([], [])))

        while not self.queue.empty():
            self.branch_step(draw)

        return self.best_soln

    def branch_step(self, draw=False):
        model_copy = self.lp.copy()
        branches = self.queue.get()[1]
        for (i, var) in enumerate(branches[0]):
            print("var: ", var)
            model_var = model_copy.getVarByName(var.VarName)
            model_copy.addConstr(model_var, grb.GRB.EQUAL, branches[1][i])
        model_copy.optimize()  # TODO: should actually get this from the queue

        if draw:
            edge_solution = grb.tupledict([(index, model_copy.getVarByName(self.x[index].VarName).X)
                                           for index in self.x.keys()])
            drawsolution.draw(self.graph, edge_solution)

        soln = {index: model_copy.getVarByName(self.x[index].VarName) for index in self.x.keys()}
        print("soln: ", soln)
        branch_var = self.branch_rule(model_copy, self.graph, soln)
        print(branch_var)

        if branch_var is None:
            return

        lp0 = model_copy.copy()
        lp1 = model_copy.copy()

        var0 = lp0.getVarByName(branch_var.VarName)
        lp0.addConstr(var0, grb.GRB.EQUAL, 0)
        branches0 = (branches[0] + [self.lp.getVarByName(branch_var.VarName)], branches[1] + [0])

        var1 = lp1.getVarByName(branch_var.VarName)
        lp1.addConstr(var1, grb.GRB.EQUAL, 1)
        branches1 = (branches[0] + [self.lp.getVarByName(branch_var.VarName)], branches[1] + [1])

        # TODO: add getting new cutting planes. Also will have to get LP solution value from external function to not
        #   waste computation

        lp0.optimize()
        lp1.optimize()

        x_vars0 = lp0.getVars()
        x_vars1 = lp1.getVars()

        if all([x.X % 1 == 0 for x in x_vars0]) and lp0.objVal < self.best_soln[0]:
            self.best_soln = (lp0.objVal, grb.tupledict([(index, lp0.getVarByName(self.x[index].VarName))
                                                         for index in self.x.keys()]))
        elif lp0.objVal < self.best_soln[0]:
            self.queue.put((lp0.objVal, branches0))

        if all([x.X % 1 == 0 for x in x_vars1]) and lp1.objVal < self.best_soln[0]:
            self.best_soln = (lp1.objVal, grb.tupledict([(index, lp1.getVarByName(self.x[index].VarName))
                                                         for index in self.x.keys()]))
        elif lp1.objVal < self.best_soln[0]:
            self.queue.put((lp1.objVal, branches1))

    def dfs_until_solution(self):
        # TODO
        pass
