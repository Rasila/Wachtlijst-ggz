# -*- coding: utf-8 -*-
"""
Functies om wachttijd te voorspellen met verschillende methodes. 
De functie werkelijke_wachttijd() bepaalt de 'werkelijke' wachttijd (het resultaat van een simulatie). 
De overige methodes kunnen aan de hand hiervan vergeleken worden. 
@author: Rasila Hoek
"""
# ----------------------------------------------------------------------------------
# Import modules
import matplotlib.pyplot as plt
import numpy as np

# Helper functions
def flatten(l):
    return [item for sublist in l for item in sublist]
# --------------------------------------------------------------------------------------
def werkelijke_wachttijd(t, wachttijden, aanmeldmomenten, num_tijdstap):
    """
    Geeft voor elke trial de wachttijd van de eerstvolgende persoon die zich aanmeldt vanaf tijdstip t.
    Als meerdere mensen zich aanmelden op tijdstip t, geeft gemiddelde wachttijd van deze mensen.
    Args:
        t = float, moment waarvoor wachttijd bepaald moet worden
        wachttijden = list of lists, element [i][j] is wachttijd van j-de starter in trial i. 
        aanmeldmomenten = List of lists, element [i][j] is aanmeldmoment van j-de starter in trial i.
        num_tijdstap = int, hoe lang liep de simulatie?
    Returns: list, met voor elke trial de wachttijd op tijdstip t
    """
    # Als t te dicht bij het einde van de simulatie zit, geef melding
    max_am = num_tijdstap - max(flatten(wachttijden))
    if(t > max_am):
        print("Dit aanmeldmoment ligt zo dicht bij het eind van de simulatie, dat de wachttijd niet goed bepaald kan worden.")
    else:
        # Initaliseer lijst voor het resultaat
        wachttijd_t = []
        # Ga alle trials af
        for trial in range(len(wachttijden)):
            # Maak arrays van aanmeldmomenten en wachttijden van deze trial
            am_trial = np.array(aanmeldmomenten[trial])
            wt_trial = np.array(wachttijden[trial])
            # Verzamel wachttijden van alle aanmelders op tijdstip t 
            # Voeg het gemiddelde hiervan toe aan lijst
            # of als er geen aanmelders zijn, probeer het tijdstip erna
            for i in range(t, num_tijdstap):
                wachttijden_i = wt_trial[am_trial == i]
                if(len(wachttijden_i) > 0):
                    wachttijd_t.append(np.mean(wachttijden_i))
                    break
            # Als het niet is gelukt, voeg een NaN toe
            if(len(wachttijd_t) < trial + 1):
                wachttijd_t.append(np.NaN)
        return wachttijd_t

def historische_wachttijd(t, delta, wachttijden, aanmeldmomenten):
    """
    Bepaalt historische wachttijd op basis van de clienten die
    in de afgelopen delta weken gestart zijn.
    Args: 
        t = float, moment waarvoor wachttijd bepaald moet worden
        delta = float, lengte van het tijdvak waarin gestarte cliënten worden meegenomen
        wachttijden = list of lists, element [i][j] is wachttijd van j-de starter in trial i. 
        aanmeldmomenten = List of lists, element [i][j] is aanmeldmoment van j-de starter in trial i. 
    Returns: list, historische wachttijd per trial
    """
    # Als delta > t, geef foutmelding
    if(delta > t):
        print("Er kan geen historische wachttijd bepaald worden, omdat het opgegeven tijdvak te lang is. ")
    else: 
        # Haal mensen die al aan het begin vd simulatie op de wl stonden eruit
        # Deze zijn niet representatief
        wachttijden_adj = []
        aanmeldmomenten_adj = []
        # ga alle trials af
        for i in range(len(wachttijden)):
            wachttijden_adj.append([])
            aanmeldmomenten_adj.append([])
            # ga alle wachttijden in deze trial af
            for j in range(len(wachttijden[i])):
                if not(aanmeldmomenten[i][j] == 0):
                    wachttijden_adj[i].append(wachttijden[i][j])
                    aanmeldmomenten_adj.append(aanmeldmomenten[i][j])
        # Initaliseer lijst voor het resultaat
        wachttijd_t  = []
        # Ga elke trial af
        for trial in range(len(wachttijden)):
            # Maak lijst met startmomenten (som van aanmeldmomenten + wachttijden)
            startmomenten = [sum(value) for value in zip(aanmeldmomenten[trial], wachttijden[trial])]
            # Verzamel alle wachttijden van mensen gestart in het tijdvak
            wachttijden_tijdvak = []
            for i in range(t - delta, t):
                for j in range(len(startmomenten)):
                    if(startmomenten[j] == i):
                        wachttijden_tijdvak.append(wachttijden_adj[trial][j])
            n = len(wachttijden_tijdvak)
            if(n == 0):
                gem = np.NaN
            else:
                gem = sum(wachttijden_tijdvak)/n
            wachttijd_t.append(gem)
        return wachttijd_t

def gvg_wachttijd(t, sim_wachtlijst, gem_behandelduur, max_capaciteit):
    """
    Bepaalt wachttijd op basis van het aantal cliënten op de wachtlijst, maal een
    schatting van de tijd tussen twee starters. 
    Args: 
        t = float, moment waarvoor wachttijd bepaald moet worden
        sim_wachtlijst = matrix, op positie (i,j) lengte wachtlijst in trial i, tijdstap j.
        gem_behandelduur = float, gemiddelde behandelduur
        max_capaciteit = int, aantal behandelplekken 
    Returns: list, wachttijd volgens gvg-methode per trial
    """
    # Initialiseer lijst voor resultaat
    wachttijd_t = []
    # Schatting gemiddelde tijd tussen twee starters
    tussentijd = gem_behandelduur/max_capaciteit
    # Ga alle trials af
    for trial in range(len(sim_wachtlijst)):
        # Bepaal hoe veel mensen op wachtlijst op tijdstap t
        n_wachtlijst = sim_wachtlijst[trial,t]
        # Bepaal verwachte wachttijd (0.95 is om drop-out wachtlijst mee te nemen)
        verwachte_wachttijd = (n_wachtlijst*0.95) * tussentijd
        wachttijd_t.append(verwachte_wachttijd)
    return wachttijd_t

#------------------------------------------------------------------------------
def methoden_test(delta, wachttijden, aanmeldmomenten, sim_wachtlijst, 
                  gem_behandelduur, max_capaciteit):
    """
    Vergelijkt historische methode met gvg-methode: hoe goed reproduceren ze
    de gesimuleerde wachttijden? 
    Visualiseert de schattingen van beide methoden voor één trial. 
    Geeft de gemiddelde afwijkingen van beide methoden.
    """
    # Maak matrices voor de schattingen van elke methode
    # Op plek (i,j) de schatting in trial i, tijdstap j
    num_tijdstap = len(sim_wachtlijst[0])
    num_cols = int(num_tijdstap - max(flatten(wachttijden)) - delta)
    num_trials = len(sim_wachtlijst)
    werkelijke_wt = np.zeros((num_trials, num_cols))
    historische_wt = np.zeros((num_trials, num_cols))
    gvg_wt = np.zeros((num_trials, num_cols))
    
    # En vul deze matrices met de schattingen van elke methode
    for t in range(int(num_tijdstap - max(flatten(wachttijden)) - delta)):
        werkelijke_wt[:, t] = werkelijke_wachttijd(
            delta + t, wachttijden, aanmeldmomenten, num_tijdstap)
        historische_wt[:, t] = historische_wachttijd(
            delta + t, delta, wachttijden, aanmeldmomenten)
        gvg_wt[:,t] = gvg_wachttijd(
            delta + t, sim_wachtlijst, gem_behandelduur, max_capaciteit)
    
    # Plot schatting van verschillende methodes voor één trial
    fig, ax = plt.subplots()
    ax.set_title("Schatting wachttijd verschillende rekenmethodes")
    ax.set_xlabel('Tijd in weken')
    ax.set_ylabel('Schatting wachttijd')
    x_punten = []
    for x in range(delta, int(num_tijdstap - max(flatten(wachttijden)))):
        x_punten.append(x)
    ax.plot(x_punten, werkelijke_wt[0,:], label = "Werkelijk")
    ax.plot(x_punten, historische_wt[0,:], label = "Methode historisch")
    ax.plot(x_punten, gvg_wt[0,:], label = "Methode gvg")
    ax.legend()
    
    # Gemiddelde afwijking per methode
    hist_verschil = np.nanmean(historische_wt - werkelijke_wt)
    gvg_verschil = np.nanmean(gvg_wt - werkelijke_wt)
    
    print("Gemiddelde afwijking methode historisch is " + str(np.round(hist_verschil, 1)) + " weken.")
    print("Gemiddelde afwijking methode gvg is " + str(np.round(gvg_verschil, 1)) + " weken.")
    return werkelijke_wt, historische_wt, gvg_wt

#--------------------------------------------------------------------------------    
    
        
        
        
        