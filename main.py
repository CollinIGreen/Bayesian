from pomegranate import *
import numpy
Difficulty = DiscreteDistribution({'d0': 0.6, "d1": 0.4})
Intelligence = DiscreteDistribution({'i0': 0.7, "i1": 0.3})
Grade = ConditionalProbabilityTable(
    [['i0', 'd0', 'g1', 0.3],
     ['i0', 'd1', 'g1', 0.05],
     ['i1', 'd0', 'g1', 0.9],
     ['i1', 'd1', 'g1', 0.5],
     ['i0', 'd0', 'g2', 0.4],
     ['i0', 'd1', 'g2', 0.25],
     ['i1', 'd0', 'g2', 0.08],
     ['i1', 'd1', 'g2', 0.3],
     ['i0', 'd0', 'g3', 0.3],
     ['i0', 'd1', 'g3', 0.7],
     ['i1', 'd0', 'g3', 0.02],
     ['i1', 'd1', 'g3', 0.2]], [Intelligence, Difficulty])
Letter = ConditionalProbabilityTable(
    [['g1', 'l0', 0.1],
     ['g1', 'l1', 0.9],
     ['g2', 'l0', 0.4],
     ['g2', 'l1', 0.6],
     ['g3', 'l0', 0.99],
     ['g3', 'l1', 0.01]], [Grade]
)
s1 = Node(Intelligence, name="Intelligence")
s2 = Node(Difficulty, name="Difficulty")
s3 = Node(Grade, name="Grade")
s4 = Node(Letter, name="Letter")
model = BayesianNetwork("Recomendation_Letter")
model.add_states(s1,s2,s3,s4)
model.add_edge(s1, s3)
model.add_edge(s2, s3)
model.add_edge(s3, s4)
model.bake()
pred = model.predict_proba({'Letter': 'l1'})
print(" ".join("{} {}".format(state.name, str(belief)) for state, belief in zip(model.states, pred)))