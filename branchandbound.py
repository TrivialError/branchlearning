import gurobipy as grb
import networkx as nx
import queue
import math
import drawsolution
import time
import DataPoint


# lp_initializer should generate a gurobi model and variable {index: names} in a tupledict given a graph
# branch_rule functions should take in the graph and an assignment to the variables and a gurobi model
#   and return a variable (index, name) to branch on
# branch_rule is one of "strong", "learned", "strongdata", "random" corresponding to the branch rule that will be used
# items in queue have format (LPvalue, queue#, {varindex: (varname, varvalue) tupledict for branches},
#                                              {varindex: (varname, varvalue) tupledict for LP solution})
# init_soln/best_soln are of the form (solution_obj_value, {varindex: (varname, varvalue) tupledict})
# node_lower_bound should take a gurobi model, a var_dict, and the graph, and return a solution;
#   that is, a {varindex: (varname, varvalue)} tupledict and an objVal


class BranchAndBound:
    def __init__(self, branch_rule, lp_initializer, graph, node_lower_bound, init_soln=(math.inf, {})):
        if branch_rule == "strong":
            self.branch_rule = self.strong_branching
        elif branch_rule == "strongdata":
            self.branch_rule = lambda model, g, soln: self.strong_branching(model, g, soln, data=True)
        elif branch_rule == "basic":
            self.branch_rule = self.basic_branch
        self.lp, self.var_dict = lp_initializer(graph)
        self.lp.params.outputflag = 0
        self.graph = graph
        self.node_lower_bound = node_lower_bound
        self.queue = queue.PriorityQueue()
        self.num_branch_nodes = 0

        self.best_soln = init_soln

    def solve(self, draw=False):
        print("Running Branch and Bound")

        lp_copy = self.lp.copy()  # not doing this copy is equivalent to keeping cutting planes around

        obj, lp_soln = self.node_lower_bound(lp_copy, self.var_dict, self.graph)

        if all([var[1] % 1 == 0 for _, var in lp_soln.items()]) and obj < self.best_soln[0]:
            self.best_soln = (obj, lp_soln)
            print("Updated best solution to: ", obj)

        self.num_branch_nodes += 1
        self.queue.put((obj, self.num_branch_nodes, grb.tupledict(), lp_soln))

        while not self.queue.empty():
            self.branch_step(draw)

        return self.best_soln

    def branch_step(self, draw=False):
        model_copy = self.lp.copy()
        queue_node = self.queue.get()
        lp_value = queue_node[0]
        # print("lp_value compared to best so far: ", lp_value, self.best_soln[0])
        if lp_value > self.best_soln[0]:
            print("branch node " + str(queue_node[1]) + " discarded directly from queue")
            return
        print("processing branch node: ", queue_node[1])
        branches = queue_node[2]
        soln = queue_node[3]
        for index, var in branches.items():
            model_var = model_copy.getVarByName(var[0])
            # print("adding branch constraint: ", model_var, var[1])
            model_copy.addConstr(model_var == var[1])

        if draw:
            edge_solution = grb.tupledict([(index, soln[index][1]) for index in soln.keys()])
            drawsolution.draw(self.graph, edge_solution)

        model_copy.update()

        branch_var = self.branch_rule(model_copy, self.graph, (lp_value, soln))
        print("branch_var: ", branch_var)

        if branch_var is None:
            return

        model_copy.update()

        lp0 = model_copy.copy()
        lp1 = model_copy.copy()

        var0 = lp0.getVarByName(branch_var[1])
        lp0.addConstr(var0 == 0)
        branches0 = dict(branches)
        branches0[branch_var[0]] = (branch_var[1], 0)

        var1 = lp1.getVarByName(branch_var[1])
        lp1.addConstr(var1 == 1)
        branches1 = dict(branches)
        branches1[branch_var[0]] = (branch_var[1], 1)

        lp0.update()
        lp1.update()

        # print("lp0 model constraint RHS: ", list(map(lambda x: x.getAttr('RHS'), lp0.getConstrs())))
        # print("lp1 model constraint RHS: ", list(map(lambda x: x.getAttr('RHS'), lp1.getConstrs())))

        a = time.clock()
        obj0, lp0_soln = self.node_lower_bound(lp0, self.var_dict, self.graph)
        obj1, lp1_soln = self.node_lower_bound(lp1, self.var_dict, self.graph)
        print("Time to solve new LPs: ", time.clock() - a)
        print("new lp objective values: ", lp0.objVal, lp1.objVal)

        if all([var[1] % 1 == 0 for _, var in lp0_soln.items()]) and obj0 < self.best_soln[0]:
            self.best_soln = (obj0, lp0_soln)
            print("Updated best solution to: ", obj0)
        elif obj0 < self.best_soln[0]:
            self.num_branch_nodes += 1
            print("adding branch node: ", self.num_branch_nodes)
            self.queue.put((obj0, self.num_branch_nodes, branches0, lp0_soln))
            # print("fractional variables when adding to queue: ",
            #       [(index, var[0], var[1]) for index, var in lp1_soln.items() if 0 < var[1] < 1])
            # print("queue node and branches for insert: ", self.num_branch_nodes, branches1)
        else:
            print("Discarding solution; obj value compared to best solution: ", obj0, self.best_soln[0])

        if all([var[1] % 1 == 0 for _, var in lp1_soln.items()]) and obj1 < self.best_soln[0]:
            self.best_soln = (obj1, lp1_soln)
            print("Updated best solution to: ", obj1)
        elif obj1 < self.best_soln[0]:
            self.num_branch_nodes += 1
            print("adding branch node: ", self.num_branch_nodes)
            self.queue.put((obj1, self.num_branch_nodes, branches1, lp1_soln))
            # print("fractional variables when adding to queue: ",
            #       [(index, var[0], var[1]) for index, var in lp1_soln.items() if 0 < var[1] < 1])
            # print("queue node and branches for insert: ", self.num_branch_nodes, branches1)
        else:
            print("Discarding solution; obj value compared to best solution: ", obj1, self.best_soln[0])

    # TODO add SB data collection
    def strong_branching(self, model, graph, soln_value, data=False):
        frac_vars = [(index, var[0]) for index, var in soln_value[1].items() if 0 < var[1] < 1]
        print("Number of variables to calculate SB score: ", len(frac_vars))
        obj = soln_value[0]
        sb_scores = {}
        for branch_var in frac_vars:
            lp0 = model.copy()
            lp1 = model.copy()

            var0 = lp0.getVarByName(branch_var[1])
            lp0.addConstr(var0, grb.GRB.EQUAL, 0)

            var1 = lp1.getVarByName(branch_var[1])
            lp1.addConstr(var1, grb.GRB.EQUAL, 1)

            lp0.update()
            lp1.update()

            obj0, lp0_soln = self.node_lower_bound(lp0, self.var_dict, self.graph)
            obj1, lp1_soln = self.node_lower_bound(lp1, self.var_dict, self.graph)

            # TODO change SB scores to 0 or 1 based on percentile
            sb_score = abs(obj - obj0)*abs(obj - obj1)/(obj**2)
            sb_scores[sb_score] = branch_var

        if not sb_scores:
            return None

        print("strong branching scores: ", sb_scores)
        if data:
            lp_solution_values = {index: var[1] for index, var in soln_value[1].items()}
            nx.set_edge_attributes(self.graph, lp_solution_values, name='solution')
            lp_soln = nx.to_numpy_matrix(self.graph, weight='solution')
            soln_adj_mat = lp_soln[lp_soln > 0] = 1
            adj_mat = nx.to_numpy_matrix(self.graph, weight='')
            weight_mat = nx.to_numpy_matrix(self.graph, weight='weight')
            for score, var in sb_scores.items():
                data = DataPoint.DataPoint(len(self.graph), lp_soln, soln_adj_mat, adj_mat, weight_mat, var[0], score)
                data.save()

        best_var = sb_scores[max(sb_scores, key=sb_scores.get)]

        return best_var

    @staticmethod
    def basic_branch(model, graph, soln_value):
        return next(((index, soln_value[1][index][0]) for index in soln_value[1].keys()
                     if 0 < soln_value[1][index][1] < 1), None)
