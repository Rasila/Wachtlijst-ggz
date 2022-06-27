# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:22:07 2022

@author: rasil
"""

# ---------------------------------------------------------------------------
# Bereken theoretische waarde van evenwichtswachttijd M/G/c queue
# Hier wordt drop-out niet in meegenomen
 
def B(c, rho):
    """
    Hulpfunctie die 'blocking probability' bepaalt. 
    Nodig in functie theoretische_wachttijd. 
    """
    if(c == 0):
        return 1
    else:
        return (rho*B(c-1, rho))/(c + rho*B(c-1, rho))
    
    
def theoretische_wachttijd(instroom, gem_behandelduur, max_capaciteit, spreiding_duur):
    """
    Bepaalt theoretisch verwachte wachttijd voor M/G/c queue.
    instroom = gemiddelde instroom per tijdseenheid
    gem_behandelduur = gemiddelde behandelduur
    max_capaciteit = aantal behandelplekken
    spreiding_duur = coefficient of variance van verdeling departure times
    """
    # Vertaal input naar handige parameters
    lamda = instroom
    mu = 1/gem_behandelduur
    c = max_capaciteit
    rho = lamda/(c*mu)
    
    # Geef melding als rho > 1
    if(rho > 1):
        print("Instroom is structureel groter dan uitstroom. De wachttijd zal blijven toenemen.")
    else:
        # Bepaal pi_w (delay probability, nodig in formule voor theoretische wachttijd)
        pi_w = (rho*B(c-1, c*rho))/(1 - rho + rho*B(c-1, c*rho))
        
        # Formule voor theoretisch verwachte wachttijd
        EW = (spreiding_duur**2 + 1)/2 * pi_w * 1/(1-rho) * 1/(c*mu)
        
        # Return resultaat
        return EW
#------------------------------------------------------------------------------

def rho_met_dropout(instroom, gem_behandelduur, max_capaciteit, p_dropout_w, 
                    p_dropout_b):
    
    gem_duur_dropout = (1-p_dropout_b)*gem_behandelduur + p_dropout_b*gem_behandelduur/2
    teller = instroom * (1-p_dropout_w)
    noemer = max_capaciteit * 1/gem_duur_dropout
    rho = teller/noemer
    
    return rho
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    