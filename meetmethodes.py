# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:16:45 2022
Functies om wachttijden te berekenen met verschillende methodes. 
Geschikt om te gebruiken op data van één simulatie. 
werkelijke_wachttijd() bepaalt de echte wachttijd. 
De overige methodes kunnen aan de hand hiervan vergeleken worden. 
@author: rasil
"""
import matplotlib.pyplot as plt
import numpy as np

def werkelijke_wachttijd(t, wachttijden, aanmeldmomenten, num_tijdstap):
    """
    Geeft de wachttijd van de eerstvolgende persoon die zich aanmeldt vanaf tijdstip t.
    Als meerdere mensen zich aanmelden op tijdstip t, geef gemiddelde wachttijd. 
    
    """
    # Als t te dicht bij het einde van de simulatie zit, geef melding
    max_am = num_tijdstap - max(wachttijden)
    if(t > max_am):
        print("Dit aanmeldmoment ligt zo dicht bij het eind van de simulatie, dat de wachttijd niet goed bepaald kan worden.")
    else:
        # Maak arrays van aanmeldmomenten en wachttijden
        aanmeldmomenten = np.array(aanmeldmomenten)
        wachttijden = np.array(wachttijden)
        # Verzamel wachttijden van alle aanmelders op tijdstip t 
        # Return het gemiddelde hiervan 
        # of als er geen aanmelders zijn, probeer het tijdstip erna
        for i in range(t, max_am):
            wachttijden_i = wachttijden[aanmeldmomenten == i]
            if(len(wachttijden_i > 0)):
                return np.mean(wachttijden_i)

def historische_wachttijd(t, delta, wachttijden, aanmeldmomenten):
    """
    Bepaalt wachttijd en spreiding van wachttijd op basis van de clienten die
    in de afgelopen weken gestart zijn. 
    t = float, moment waarvoor wachttijd bepaald moet worden
    delta = float, lengte van het tijdvak waarin gestarte cliënten worden meegenomen
    wachttijden = list, lijst met wachttijden uit simulatie
    aanmeldmomenten = list, lijst met bijbehorende aanmeldmomenten uit simulatie 
    
    returns: geschatte wachttijd en standaardafwijking
    """
    # Als delta > t, geef foutmelding
    if(delta > t):
        print("Er kan geen historische wachttijd bepaald worden, omdat het opgegeven tijdvak te lang is. ")
    else: 
        # Maak lijst met startmomenten (som van aanmeldmomenten + wachttijden)
        startmomenten = [sum(value) for value in zip(aanmeldmomenten, wachttijden)]
        # Verzamel alle wachttijden van mensen gestart in het tijdvak
        wachttijden_tijdvak = []
        for i in range(t - delta, t):
            for j in range(len(startmomenten)):
                if(startmomenten[j] == i):
                    wachttijden_tijdvak.append(wachttijden[j])
        n = len(wachttijden_tijdvak)
        gem = sum(wachttijden_tijdvak)/n
        # print("Wachttijd bepaald op basis van " + str(n) + " clienten.")
        return gem
                    
def gvg_wachttijd(t, delta, wachttijden, aanmeldmomenten, sim_wachtlijst):
    if(delta > t):
        print("Er kan geen wachttijd bepaald worden, omdat het opgegeven tijdvak te lang is. ")
    else: 
        # Maak lijst met startmomenten (som van aanmeldmomenten + wachttijden)
        startmomenten = [sum(value) for value in zip(aanmeldmomenten, wachttijden)]
        # Bepaal het aantal simulaties
        num_sim = len(sim_wachtlijst)
        # Bepaal hoe veel mensen in het tijdvak zijn gestart
        n = 0
        for i in range(t - delta, t):
            for j in range(len(startmomenten)):
                if(startmomenten[j] == i):
                    n += 1
        # Bepaal gemiddelde tijd tussen twee starters
        tussentijd = delta*num_sim/n
        # Bepaal gem aantal mensen op wachtlijst in tijdstap t
        n_wachtlijst = sum(sim_wachtlijst[:,t])/num_sim
        # Bepaal verwachte wachttijd
        verwachte_wachttijd = n_wachtlijst * 0.9 * tussentijd
        return verwachte_wachttijd

def gvg2_wachttijd(t, sim_wachtlijst, gem_behandelduur, max_capaciteit):
    # Bepaal het aantal simulaties (dit moet eigenlijk altijd 1 zijn)
    num_sim = len(sim_wachtlijst)
    # Schatting gemiddelde tijd tussen twee starters
    tussentijd = gem_behandelduur/max_capaciteit
    # Bepaal hoe veel mensen op wachtlijst op tijdstap t
    n_wachtlijst = sum(sim_wachtlijst[:,t])/num_sim
    # Bepaal verwachte wachttijd (0.95 is om drop-out wachtlijst mee te nemen)
    verwachte_wachttijd = (n_wachtlijst*0.9) * tussentijd
    return verwachte_wachttijd

#------------------------------------------------------------------------------
# TEST verschillende methoden
def methoden_test(delta, wachttijden, aanmeldmomenten, sim_wachtlijst, 
                  gem_behandelduur, max_capaciteit):
    
    werkelijke_wt = []
    historische_wt = []
    gvg_wt = []
    gvg2_wt = []
    num_tijdstap = len(sim_wachtlijst[0])
    for t in range(delta, int(num_tijdstap - max(wachttijden) - 0.5*delta)):
        werkelijke_wt.append(werkelijke_wachttijd(
            t, wachttijden, aanmeldmomenten, num_tijdstap))
        historische_wt.append(historische_wachttijd(
            t, delta, wachttijden, aanmeldmomenten))
        gvg_wt.append(gvg_wachttijd(
           t, delta, wachttijden, aanmeldmomenten, sim_wachtlijst))
        gvg2_wt.append(gvg2_wachttijd(
            t, sim_wachtlijst, gem_behandelduur, max_capaciteit))
    
    # Plot schatting van verschillende methodes
    fig, ax = plt.subplots()
    ax.set_title("Schatting wachttijd verschillende rekenmethodes")
    ax.set_xlabel('Tijd in weken')
    ax.set_ylabel('Schatting wachttijd')
    x_punten = []
    for x in range(delta, int(num_tijdstap - max(wachttijden) - 0.5*delta)):
        x_punten.append(x)
    ax.plot(x_punten, werkelijke_wt, label = "Werkelijk")
    ax.plot(x_punten, historische_wt, label = "Methode historisch")
    #ax.plot(x_punten, gvg_wt, label = "Methode gvg")
    ax.plot(x_punten, gvg2_wt, label = "Methode gvg 2")
    ax.legend()
    
    # Bereken afwijking met werkelijke wachttijd
    verschil_hist = np.array(historische_wt) - np.array(werkelijke_wt)
    # verschil_gvg = np.array(gvg_wt) - np.array(werkelijke_wt)
    verschil_gvg2 = np.array(gvg2_wt) - np.array(werkelijke_wt)
    
    return verschil_hist, verschil_gvg2
    
    
    
        
        
        
        