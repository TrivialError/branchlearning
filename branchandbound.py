import gurobipy as grb
import networkx as nx
import queue
import math
import drawsolution
import time
import random
import TSPfunctions
import trained_model_interface
from SBScoresData import *


# lp_initializer should generate a gurobi model and variable {index: names} in a tupledict given a graph
# branch_rule functions should take in the graph and an assignment to the variables and a gurobi model
#   and return a variable (index, name) to branch on
# branch_rule is one of "strong", "learned", "strongdata", "random" corresponding to the branch rule that will be used
# items in queue have format (LPvalue, queue#, {varindex: (varname, varvalue) tupledict for branches},
#                                              {varindex: (varname, varvalue) tupledict for LP solution},
#                                              [cutting_plane gurobi constraints])
# init_soln/best_soln are of the form (solution_obj_value, {varindex: (varname, varvalue) tupledict})
# node_lower_bound should take a gurobi model, a var_dict, and the graph, and return a solution;
#   that is, a {varindex: (varname, varvalue)} tupledict and an objVal


class BranchAndBound:
    def __init__(self, tsp_instance, branch_rule, lp_initializer, graph, node_lower_bound, init_soln=(math.inf, {})):
        self.tsp_instance = tsp_instance
        if branch_rule == "strong":
            self.branch_rule = self.strong_branching
        elif branch_rule == "strongdata":
            self.branch_rule = lambda model, g, soln: self.strong_branching(model, g, soln, data=True)
        elif branch_rule == "basic":
            self.branch_rule = self.basic_branch
        elif branch_rule == "random":
            self.branch_rule = self.random_branch
        self.lp, self.var_dict = lp_initializer(graph)
        self.lp.params.outputflag = 0
        self.graph = graph
        self.node_lower_bound = node_lower_bound
        self.queue = queue.PriorityQueue()
        self.num_branch_nodes = 0
        self.best_soln = init_soln

    def solve(self, draw=False, timeout=math.inf):
        print("Running Branch and Bound")
        t = time.clock()

        lp_copy = self.lp.copy()

        obj, lp_soln, new_constrs = self.node_lower_bound(lp_copy, self.var_dict, self.graph)
        if obj is None:
            print("Initial problem is infeasible")
            return

        if draw:
            edge_solution = grb.tupledict([(index, lp_soln[index][1]) for index in lp_soln.keys()])
            drawsolution.draw(self.graph, edge_solution)

        if all([var[1] % 1 == 0 for _, var in lp_soln.items()]) and obj < self.best_soln[0]:
            self.best_soln = (obj, lp_soln)
            print("solution found without branching; value: ", obj)
            return self.best_soln, 0

        self.num_branch_nodes += 1
        self.queue.put((obj, self.num_branch_nodes, grb.tupledict(), lp_soln, new_constrs))

        while not self.queue.empty():
            self.branch_step(draw)
            if time.clock() - t > timeout:
                return None, None

        if draw:
            edge_solution = grb.tupledict([(index, self.best_soln[1][index][1]) for index in self.best_soln[1].keys()])
            drawsolution.draw(self.graph, edge_solution)

        return self.best_soln, self.num_branch_nodes

    def branch_step(self, draw=False):
        model_copy = self.lp.copy()
        queue_node = self.queue.get()
        lp_value = queue_node[0]
        if lp_value > self.best_soln[0]:
            print("branch node " + str(queue_node[1]) + " discarded directly from queue")
            return
        print("processing branch node: ", queue_node[1])
        print("lp_value compared to best so far: ", lp_value, self.best_soln[0])
        branches = queue_node[2]
        soln = queue_node[3]
        extra_constrs = queue_node[4]
        for index, var in branches.items():
            model_var = model_copy.getVarByName(var[0])
            # print("adding branch constraint: ", model_var, var[1])
            model_copy.addConstr(model_var == var[1])

        print("number of extra_constrs: ", len(extra_constrs))
        # uncomment the following to add cutting plane saving
        # for constr in extra_constrs:
        #     # TODO technically this shouldn't rely on TSPfunctions at all
        #     TSPfunctions.tsp_get_constrs_from_description(model_copy, constr)

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
        obj0, lp0_soln, new_constrs0 = self.node_lower_bound(lp0, self.var_dict, self.graph)
        obj1, lp1_soln, new_constrs1 = self.node_lower_bound(lp1, self.var_dict, self.graph)
        # print("new_constrs0: ", len(new_constrs0))
        # print("new_constrs1: ", len(new_constrs1))
        print("Time to solve new LPs: ", time.clock() - a)
        print("new lp objective values: ", obj0, obj1)

        if obj0 is not None:
            if all([var[1] % 1 == 0 for _, var in lp0_soln.items()]) and obj0 < self.best_soln[0]:
                self.best_soln = (obj0, lp0_soln)
                print("Updating best solution to: ", obj0)
            elif obj0 < self.best_soln[0]:
                self.num_branch_nodes += 1
                print("adding branch node: ", self.num_branch_nodes)
                self.queue.put((obj0, self.num_branch_nodes, branches0, lp0_soln, extra_constrs + new_constrs0))
                # print("fractional variables when adding to queue: ",
                #       [(index, var[0], var[1]) for index, var in lp1_soln.items() if 0 < var[1] < 1])
                # print("queue node and branches for insert: ", self.num_branch_nodes, branches1)
            else:
                print("Discarding solution; obj value compared to best solution: ", obj0, self.best_soln[0])

        if obj1 is not None:
            if all([var[1] % 1 == 0 for _, var in lp1_soln.items()]) and obj1 < self.best_soln[0]:
                self.best_soln = (obj1, lp1_soln)
                print("Updating best solution to: ", obj1)
            elif obj1 < self.best_soln[0]:
                self.num_branch_nodes += 1
                print("adding branch node: ", self.num_branch_nodes)
                self.queue.put((obj1, self.num_branch_nodes, branches1, lp1_soln, extra_constrs + new_constrs1))
                # print("fractional variables when adding to queue: ",
                #       [(index, var[0], var[1]) for index, var in lp1_soln.items() if 0 < var[1] < 1])
                # print("queue node and branches for insert: ", self.num_branch_nodes, branches1)
            else:
                print("Discarding solution; obj value compared to best solution: ", obj1, self.best_soln[0])

    def strong_branching(self, model, graph, soln_value, data=False, alpha=0.2):
        frac_vars = [(index, var[0]) for index, var in soln_value[1].items() if 0 < var[1] < 1]
        print("Number of variables to calculate SB score: ", len(frac_vars))
        obj = soln_value[0]
        sb_scores = []
        for branch_var in frac_vars:
            lp0 = model.copy()
            lp1 = model.copy()

            var0 = lp0.getVarByName(branch_var[1])
            lp0.addConstr(var0, grb.GRB.EQUAL, 0)

            var1 = lp1.getVarByName(branch_var[1])
            lp1.addConstr(var1, grb.GRB.EQUAL, 1)

            lp0.update()
            lp1.update()

            obj0, lp0_soln, _ = self.node_lower_bound(lp0, self.var_dict, self.graph)
            obj1, lp1_soln, _ = self.node_lower_bound(lp1, self.var_dict, self.graph)

            if obj0 is None or obj1 is None:
                sb_score = math.inf
            else:
                sb_score = max(0.1, obj0 - obj) * max(0.1, obj1 - obj)
            sb_scores.append((sb_score, branch_var))
            if not data and obj0 and obj1:
                if obj0 > self.best_soln[0] and obj1 > self.best_soln[0]:
                    print("ending SB early; found trimmable branches")
                    break

        if not sb_scores:
            return None

        sb_scores_sorted = sorted(sb_scores, key=lambda item: item[0], reverse=True)

        best_score = sb_scores_sorted[0][0]
        print("best branch score: ", best_score)

        if data and len(sb_scores) >= 4:
            best_score = sb_scores_sorted[0][0]
            sb_scores_labels = [(1, var) if score >= (1 - alpha) * best_score else (0, var)
                                for (score, var) in sb_scores]
            sb_scores_labels0 = [(0, var) for score, var in sb_scores_labels if score == 0]
            sb_scores_labels1 = [(1, var) for score, var in sb_scores_labels if score == 1]
            if len(sb_scores_labels0) != 0:
                if len(sb_scores_labels1) / len(sb_scores_labels0) < 0.2:
                    t = (len(sb_scores_labels1)/(0.3*len(sb_scores_labels0)))  # portion of 0-labels to keep
                    if t < 1:
                        print("Deleting " + str(int(round((1 - t) * len(sb_scores_labels0)))) + " data label zeros")
                        sb_scores_labels0 = random.sample(sb_scores_labels0, round(t*len(sb_scores_labels0)))
            sb_scores_labels = sb_scores_labels0 + sb_scores_labels1
            lp_solution_values = {index: var[1] for index, var in soln_value[1].items()}
            nx.set_edge_attributes(self.graph, lp_solution_values, name='solution')
            lp_soln = nx.to_numpy_matrix(self.graph, weight='solution')
            soln_adj_mat = lp_soln.copy()
            soln_adj_mat[soln_adj_mat > 0] = 1
            adj_mat = nx.to_numpy_matrix(self.graph, weight='')
            weight_mat = nx.to_numpy_matrix(self.graph, weight='weight')
            var_sb_label_dict = {var[0]: sb_label for sb_label, var in sb_scores_labels}
            data = SBScoresData(self.tsp_instance, len(self.graph), lp_soln, soln_adj_mat,
                                adj_mat, weight_mat, var_sb_label_dict)
            data.save()

        best_var = sb_scores_sorted[0][1]

        return best_var

    @staticmethod
    def basic_branch(model, graph, soln_value):
        return next(((index, soln_value[1][index][0]) for index in soln_value[1].keys()
                     if 0 < soln_value[1][index][1] < 1), None)

    @staticmethod
    def random_branch(model, graph, soln_value):
        return random.choice([(index, soln_value[1][index][0]) for index in soln_value[1].keys()
                              if 0 < soln_value[1][index][1] < 1])

    def learned_branch(self, model, graph, soln_value):
        lp_solution_values = {index: var[1] for index, var in soln_value[1].items()}
        nx.set_edge_attributes(self.graph, lp_solution_values, name='solution')
        lp_soln = nx.to_numpy_matrix(self.graph, weight='solution')
        soln_adj_mat = lp_soln.copy()
        soln_adj_mat[soln_adj_mat > 0] = 1
        adj_mat = nx.to_numpy_matrix(self.graph, weight='')
        weight_mat = nx.to_numpy_matrix(self.graph, weight='weight')
        data = SBScoresData(self.tsp_instance, len(self.graph), lp_soln, soln_adj_mat,
                            adj_mat, weight_mat, {}, train=False)
        branch_var_labels = trained_model_interface.get_branch_var_labels(data)
        branch_var = max(branch_var_labels, key=lambda bvl: bvl[1])[0]
        return branch_var, self.var_dict[branch_var]
