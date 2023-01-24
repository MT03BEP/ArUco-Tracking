"""
In dit script staan de functies voor het berekenen van snelheden en afgeleiden. Dit script 
wordt aangeroepen door de main code.
"""

# Berekent de afgeleide tov t en een nieuwe tijd array voor plotten
def CalculateDerivative(x,t):
    assert (len(x)==len(t))
    derivative = (x[1:] - x[:-1]) / (t[:-1] - t[1:])
    new_time_vec = t[1:]
    return derivative, new_time_vec