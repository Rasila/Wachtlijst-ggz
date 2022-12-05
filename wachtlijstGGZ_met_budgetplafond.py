# -*- coding: utf-8 -*-
"""
@author: Rasila Hoek
"""

# -*- coding: utf-8 -*-
"""
Agent-based model voor wachtlijst ggz, met budgetplafonds per zorgverzekeraar.

Instroom gemodelleerd met een Poisson-proces 
(parameter: gemidddelde instroom). 
Behandelduur gemodelleerd met een normale verdeling 
(parameters: gemiddelde behandelduur, spreiding behandelduur)
Clienten kunnen drop-out gaan aan het eind van de hun wachttijd,
of gedurende de behandeling.
(parameters: kans drop-out wachtlijst, kans drop-out behandeling)

Brengt in beeld hoe de aantallen mensen op de wachtlijst en in behandeling verlopen.
Laat zien hoe de verdeling van wachttijden en het verloop van wachttijd over de tijd 
eruit ziet voor cliënten van verschillende zorgverzekeraars.  
"""
# ----------------------------------------------------------------------------
# Import modules
import numpy as np
import math
import matplotlib.pyplot as plt
import random

# Helper functions
def flatten(l):
    return [item for sublist in l for item in sublist]
# ----------------------------------------------------------------------------
class client(object):
    
    """
    attributes:
        rest_behandelduur: float, de resterende behandelduur van deze client 
        wachttijd: int, hoe lang heeft deze cliënt tot nu toe gewacht?
        zv: zorgverzekeraar van deze cliënt
    """
    
    def __init__(self, rest_behandelduur, wachttijd, zv):
        # Set attributes
        self.rest_behandelduur = rest_behandelduur
        self.wachttijd = wachttijd
        self.zv = zv
    def update_behandelduur(self):
        "Update van de resterende behandelduur"
        self.rest_behandelduur = self.rest_behandelduur - 1
    def update_wachttijd(self):
        "Houdt bij hoe lang client al gewacht heeft"
        self.wachttijd = self.wachttijd + 1

class behandeling(object):
    
    """
    attributes:
        wachtlijst: list, cliënten op de wachtlijst
        in_behandeling: list, cliënten in behandeling
        max_capaciteit: int, totaal aantal behandelplekken
        gem_behandelduur: int, gemiddelde behandelduur in weken
        tijd: int, hoe veel weken loopt de simulatie al
        plafonds: dict, key is zorgverzekeraar, value is resterende behandelingen dit jaar
        
    """
    def __init__(self, wachtlijst, in_behandeling, 
                 max_capaciteit, gem_behandelduur, plafonds, tijd):

        # Set attributes
        self.wachtlijst = wachtlijst
        self.in_behandeling = in_behandeling
        self.max_capaciteit = max_capaciteit
        self.gem_behandelduur = gem_behandelduur
        self.plafonds = plafonds
        self.tijd = tijd
        
    def update(self, instroom, p_dropout_w, p_dropout_b, spreiding_duur, zv_kansen):
        """
        Functie die wachtlijst en in_behandeling updatet. 
        Zorgt voor instroom wachtlijst op basis van Poisson met parameter instroom.
        Checkt of er behandelplekken vrij zijn en laat wachtenden instromen.
        Checkt of er cliënten klaar zijn om uit te stromen. 
        Checkt of er cliënten drop-out gaan.
        Updatet de resterende plekken per zorgverzekeraar.
        Returns: wachttijden (list), aanmeldmomenten (list), zorgverzekeraars (list) van gestarte cliënten
        """
        
        # TIJD BIJHOUDEN
        self.tijd = self.tijd + 1
        
        # INSTROOM WACHTLIJST op basis van Poisson met parameter instroom.
        aantal_in = np.random.poisson(instroom)
        # Creëer nieuwe cliënten voor op wachtlijst
        nieuwe_clienten = []
        for i in range(aantal_in):
            # Bepaal behandelduur en verzekeraar
            behandelduur = np.random.normal(self.gem_behandelduur, spreiding_duur*self.gem_behandelduur)
            zv = random.choices(list(self.plafonds.keys()), weights = zv_kansen)
            zv = zv[0]
            nieuwe_client = client(behandelduur, 0, zv)
            nieuwe_clienten.append(nieuwe_client)
        # Voeg nieuwe clienten toe aan wachtlijst
        self.wachtlijst.extend(nieuwe_clienten)
        
        # INSTROOM BEHANDELING op basis van plekken vrij, wachtenden, drop-out en plafonds
        # Bepaal hoe veel plekken vrij
        plekken_vrij = self.max_capaciteit - len(self.in_behandeling)
        # Initialiseer lijst wachttijden, aanmeldmomenten, zorgverzekeraars
        wachttijden = []
        aanmeldmomenten = []
        zorgverzekeraars = []
        # Probeer voor elke vrije plek iemand te laten starten
        for i in range(plekken_vrij):
            # k houdt bij welke wachtende wordt overwogen
            # te beginnen met de eerste wachtende, k = 0 
            k = 0
            # zolang er nog wachtenden zijn...
            while k < len(self.wachtlijst):
                # neem persoon k in overweging als starter
                starter = self.wachtlijst[k]
                zv_starter = starter.zv
                # als hij een budgetplafond heeft...
                if self.plafonds[zv_starter] == 0:
                    # .. neem volgende persoon in overweging
                    k += 1
                # als hij geen budgetplafond heeft...
                else:
                    # als drop-out...
                    if(np.random.binomial(1, p_dropout_w)):
                        # ...verwijder van wachtlijst
                        del self.wachtlijst[k]
                    # als geen drop-out...
                    else:
                        # ..is dit de persoon die gaat starten
                        break
            # Als niemand is gevonden zonder bp en zonder drop-out, doe niets
            if len(self.wachtlijst) == k:
                break
            
            # Laat persoon starten
            if self.plafonds[zv_starter] > 0:
                self.in_behandeling.append(starter)
                # Registreer de wachttijd van deze persoon
                wachttijden.append(starter.wachttijd)
                aanmeldmomenten.append(self.tijd - starter.wachttijd)
                zorgverzekeraars.append(zv_starter)
                # En update plafonds
                self.plafonds[zv_starter] = self.plafonds[zv_starter] - 1
                # Verwijder deze persoon van wachtlijst
                del self.wachtlijst[k]
            
        # UITSTROOM DOOR AFRONDEN BEHANDELING    
        # Check voor elke client of ie nog in behandeling blijft
        # Zo niet, voeg m niet toe aan de nieuwe in_behandeling
        nieuwe_in_behandeling = []
        for clt in self.in_behandeling:
           if not(clt.rest_behandelduur < 0):
               nieuwe_in_behandeling.append(clt)
        self.in_behandeling = nieuwe_in_behandeling
        
        # DROP-OUT BEHANDELING
        # Voor elke persoon in behandeling, bepaal wel of geen drop-out
        in_behandeling_gedropt = []
        for clt in self.in_behandeling:
            if not(np.random.binomial(1, 1-(1-p_dropout_b)**(1/self.gem_behandelduur))):
                in_behandeling_gedropt.append(clt)
        self.in_behandeling = in_behandeling_gedropt
        
        # Return wachttijden, aanmeldmomenten en zorgverzekeraars
        return wachttijden, aanmeldmomenten, zorgverzekeraars
    
# ----------------------------------------------------------------------------
def simuleer_wachtlijst_bp(num_wachtlijst_start, 
                           rho_start, 
                           max_capaciteit,
                           instroom,
                           gem_behandelduur,
                           spreiding_duur,
                           p_dropout_w,
                           p_dropout_b,
                           num_trials,
                           num_tijdstap,
                           start_plafonds,
                           zv_kansen):
    """
    Functie die de simulatie met budgetplafonds runt. 
    Args:
        num_wachtlijst_start = int, aantal clienten op wachtlijst bij begin simulatie
        rho_start = float, gedeelte waarvoor de capaciteit is gevuld bij begin simulatie
        max_capaciteit = int, aantal plekken in behandeling
        instroom = float, gemiddelde wekelijkse instroom
        gem_behandelduur = float, gemiddelde behandelduur
        spreiding_duur = float, ratio sd/gemiddelde van behandelduur
        p_dropout_w = kans dat iemand op enig moment tijdens wachtlijst uitvalt
        p_dropout_b = kans dat iemand op enig moment tijdens behandeling uitvalt
        num_trials = int, hoe vaak moet de simulatie worden gedaan?
        num_tijdstap = int, hoe veel weken moet de simulatie lopen?
        start_plafonds = dictionary met key verzekeraar, value plafond
        zv_kansen = list met hoe vaak elke zorgverzekeraar voorkomt (bijv percentages)
    """
    # Initialiseer matrices/lijsten om resultaten van trials te bewaren
    start_plafonds_kopie = start_plafonds.copy()
    wachttijden = []
    aanmeldmomenten = []
    zorgverzekeraars = []
    sim_plafonds = np.zeros((num_trials, num_tijdstap, len(start_plafonds_kopie)))
    sim_wachtlijst = np.zeros((num_trials, num_tijdstap))
    sim_in_behandeling = np.zeros((num_trials, num_tijdstap))

    # Voor num_trials trials
    for trial in range(num_trials):
        
        # Initialiseer lijsten voor wachttijden, aanmeldmomenten in deze trial
        wt_trial = []
        am_trial = []
        zv_trial = []
        
        # Initialiseer wachtlijst met nieuwe cliënten
        start_wachtlijst = []
        for i in range(num_wachtlijst_start):
            behandelduur = np.random.normal(gem_behandelduur, spreiding_duur*gem_behandelduur)
            zv = random.choices(list(start_plafonds.keys()), weights = zv_kansen)
            zv = zv[0]
            start_wachtlijst.append(client(behandelduur, 0, zv))
        
        # Initialiseer in_behandeling met cliënten die al variabel lang in behandeling zijn
        start_in_behandeling = []
        for i in range(math.floor(rho_start*max_capaciteit)):
            behandelduur = np.random.normal(gem_behandelduur, spreiding_duur*gem_behandelduur)
            zv = random.choices(list(start_plafonds.keys()), weights = zv_kansen)
            zv = zv[0]
            al_behandeld = np.random.uniform(0, behandelduur)            
            start_in_behandeling.append(client(behandelduur - al_behandeld, 0, zv))
        
        # Initialiseer behandeling met deze wachtlijst, in_behandeling
        sim_behandeling = behandeling(start_wachtlijst, start_in_behandeling, 
                                      max_capaciteit, gem_behandelduur, 
                                      start_plafonds_kopie, tijd = 0)
        # Voor elke tijdstap:
        for tijdstap in range(num_tijdstap):
            # Update wachttijd voor alle clienten in wachtlijst
            for clt in sim_behandeling.wachtlijst:
                clt.update_wachttijd()
            # Update rest_behandelduur alle clienten in in_behandeling
            for clt in sim_behandeling.in_behandeling:
                clt.update_behandelduur()
            # Update de behandeling en sla wachttijden, aanmeldmomenten op
            wt, am, zv = sim_behandeling.update(instroom, p_dropout_w, 
                                                p_dropout_b, spreiding_duur, zv_kansen)
            wt_trial.extend(wt)
            am_trial.extend(am)
            zv_trial.extend(zv)
            # Als eind jaar, de plafonds weer terugzetten op startwaarde
            if(tijdstap%52 == 0):
                sim_behandeling.plafonds = start_plafonds.copy()
            
            # Bereken en bewaar resultaten 
            # Sla de overgebleven plekken per zorgverzekeraar op
            zvs = list(sim_behandeling.plafonds.keys())
            num_zvs = len(zvs)
            for i in range(num_zvs):
                sim_plafonds[trial, tijdstap, i] = sim_behandeling.plafonds[zvs[i]]
            # Sla grootte wachtlijst, in_behandeling en rho op
            sim_wachtlijst[trial, tijdstap] = len(sim_behandeling.wachtlijst)
            sim_in_behandeling[trial, tijdstap] = len(sim_behandeling.in_behandeling)
            rho = sim_in_behandeling/max_capaciteit
        # Sla wachttijden, aanmeldmomenten en zorgverzekeraars van deze trial op
        wachttijden.append(wt_trial)
        aanmeldmomenten.append(am_trial)
        zorgverzekeraars.append(zv_trial)
            
    return sim_wachtlijst, sim_in_behandeling, sim_plafonds, wachttijden, aanmeldmomenten, zorgverzekeraars, rho, max_capaciteit, zvs
# -------------------------------------------------------------------------------
def gemiddelde_wachttijd(t, delta, wachttijden, aanmeldmomenten, num_tijdstap):
    """
    Hulpfunctie voor resultaten_simulatie_bp()
    Bepaalt gemiddelde wachttijd op bepaald tijdstip over meerdere simulaties.
    Args:
        t = int, aanmeldmoment waarvoor gemiddelde wachttijd bepaald moet worden
        delta = int, lengte van het tijdvak rondom tijdstip t
        wachttijden = lijst met wachttijden uit simulatie
        aanmeldmomenten = lijst met bijbehorende aanmeldmomenten uit simulatie 
        num_tijdstap = int, hoe lang liep de simulatie?    
    Returns: gemiddelde wachttijd
    """
    # Als t te veel aan het eind van de simulatie zit, waardoor er geen volledige
    # data meer beschikbaar is, geef melding. 
    max_am = num_tijdstap - max(wachttijden)
    if(t + 0.5*delta > max_am):
        print("Dit aanmeldmoment ligt zo dicht bij het eind van de simulatie, dat de wachttijd niet goed bepaald kan worden.")
    else:
        # verzamel alle wachttijden binnen het tijdvak
        # met uitzondering van wachttijden bij aanmeldmoment t=0
        wachttijden_tijdvak = []
        for i in range(max(int(t - 0.5*delta), 1), int(t + 0.5*delta)):
           for j in range(len(aanmeldmomenten)):
               if(aanmeldmomenten[j] == i):
                   wachttijden_tijdvak.append(wachttijden[j])
        n = len(wachttijden_tijdvak)               
        if(n == 0):
            print("Geen starters in het tijdvak om gemiddelde wachttijd te bepalen")
        else:
            gem = sum(wachttijden_tijdvak)/n
            return gem
        
# -------------------------------------------------------------------------------

def resultaten_simulatie_bp(sim_wachtlijst, sim_in_behandeling, sim_plafonds, 
                            wachttijden, aanmeldmomenten, zorgverzekeraars,
                            rho, max_capaciteit, zvs):
    """
    Visualiseert het verloop van de simulatie en geeft samenvatting resultaten.
    Args:
        sim_wachtlijst: matrix, op positie (i,j) lengte wachtlijst in trial i, tijdstap j.
        sim_in_behandeling: matrix, op positie (i,j) aantal mensen in behandeling in trial i, tijdstap j. 
        sim_plafonds: array, op positie (i,j,k) de overgebleven plekken in trial i, tijdstap j, 
            zorgverzekeraar k. 
        wachttijden: List of lists, element [i][j] is wachttijd van j-de starter in trial i. 
        aanmeldmomenten: List of lists, element [i][j] is aanmeldmoment van j-de starter in trial i. 
        zorgverzekeraars: list of lists, element [i][j] is zorgverzekeraar van j-de starter in trial i. 
        rho: matrix, op positie (i,j) de mate waarin behandeling gevuld is in trial i, tijdstap j. 
        max_capaciteit: aantal behandelplekken
        zvs: geordende lijst van zorgverzekeraars
    """
    # Bepaal grootte simulatie
    num_tijdstap = len(sim_wachtlijst[0])
    num_sim = len(sim_wachtlijst)
    num_punten = len(flatten(wachttijden))
    
    # Maak aangepaste lijst met wachttijden, alleen unbiased data
    # Geen mensen die op t=0 al op wachtlijst stonden
    wachttijden_adj = []
    # ga alle trials af
    for i in range(len(wachttijden)):
        wachttijden_adj.append([])
        # ga alle wachttijden in deze trial af
        for j in range(len(wachttijden[i])):
            if(aanmeldmomenten[i][j] < num_tijdstap - max(flatten(wachttijden))):
                if not(aanmeldmomenten[i][j] == 0):
                    wachttijden_adj[i].append(wachttijden[i][j])
    
    # Enkele resultaten om te printen
    wachttijd_gem = sum(flatten(wachttijden_adj))/len(flatten(wachttijden_adj))
    rho_gem = sum(sum(rho))/np.size(rho)
    wachttijd_gem_tekst = str(round(wachttijd_gem))
    rho_gem_tekst = str(round(100*rho_gem))
    clienten_gem = num_punten/num_sim
    clienten_gem_tekst = str(round(clienten_gem))  
    print(str(num_sim) + " simulaties van " + str(num_tijdstap) + " weken.")
    print("De gemiddelde wachttijd over alle simulaties is " + wachttijd_gem_tekst + " weken.")
    print("De capaciteit was gemiddeld voor " + rho_gem_tekst + " procent gevuld.")
    print("Er zijn per simulatie gemiddeld " + clienten_gem_tekst + " clienten gestart met behandeling.")
    # Bepaal percentage dat korter moest wachten dan treeknorm
    onder_treek = 0
    for trial in range(len(wachttijden_adj)):
        for wachttijd in wachttijden_adj[trial]:
            if(wachttijd < 14):
                onder_treek = onder_treek + 1
    procent_onder_treek = round(100*onder_treek/len(flatten(wachttijden_adj)))
    print(str(procent_onder_treek) + " procent is binnen treeknorm gestart.")
    
    # VERDELING WACHTTIJDEN histogram
    fig, ax = plt.subplots()
    ax.set_title('Verdeling wachttijd in ' + str(num_sim) + ' simulaties')
    ax.hist(flatten(wachttijden_adj))
    ax.set_xlabel('Aantal weken')
    
    # WACHTTIJD PER AANMELDMOMENT scatterplot
    fig, ax = plt.subplots()
    # Titels en assen
    ax.set_title('Wachttijden per aanmeldmoment (' + str(num_sim) + ' simulaties)')
    ax.set_xlabel('Week van aanmelden')
    ax.set_ylabel('Aantal weken gewacht')

    # Maak alle lists flat
    aanmeldmomenten_f = flatten(aanmeldmomenten)
    wachttijden_f = flatten(wachttijden)
    zorgverzekeraars_f = flatten(zorgverzekeraars)
    
    # Maak scatterplot per zorgverzekeraar
    for zv in zvs:
        x_punten = []
        y_punten = []
        for i in range(num_punten):
            if(zorgverzekeraars_f[i] == zv):
                if not(aanmeldmomenten_f[i] == 0):
                    x_punten.append(aanmeldmomenten_f[i])
                    y_punten.append(wachttijden_f[i])
        ax.scatter(x_punten, y_punten, s = 1, label = zv)
    # Plot de lijn voor de treeknorm en het gemiddelde
    ax.axhline(y=14, color='r', linestyle='--', linewidth = 1, label = 'treeknorm')
    ax.axhline(y=wachttijd_gem, color='black', linestyle = '--', linewidth = 1, label = 'gemiddelde')
    # Maak legend
    ax.legend()
    # Plot eindlijn
    x_eindlijn = np.linspace(num_tijdstap - max(wachttijden_f), num_tijdstap, 100)
    y_eindlijn = num_tijdstap - x_eindlijn
    ax.fill_between(x_eindlijn, y_eindlijn, max(wachttijden_f), alpha = 0.5, color = 'black')
    
    # VERLOOP GEMIDDELDE WACHTTIJD grafiek
    fig, ax = plt.subplots()
    ax.set_title('Verloop gemiddelde wachttijd')
    ax.set_xlabel('Week van aanmelden')
    ax.set_ylabel('Wachttijd')
    delta = int(num_tijdstap/13)
    x_punten = []
    for x in range(int(num_tijdstap - max(wachttijden_f) - 0.5*delta)):
        x_punten.append(x)
    # Plot voor elke zv een lijn voor de gemiddelde wachttijd
    for zv in zvs:
        # Maak aangepaste lijsten met wachttijden en aanmeldmomenten
        # specifiek voor deze zorgverzekeraar
        wachttijden_zv = []
        aanmeldmomenten_zv = []
        for i in range(len(wachttijden_f)):
            if(zorgverzekeraars_f[i] == zv):
                wachttijden_zv.append(wachttijden_f[i])
                aanmeldmomenten_zv.append(aanmeldmomenten_f[i])
        y_punten = []
        for x in x_punten:
            y_punten.append(gemiddelde_wachttijd(
                x, delta, wachttijden_zv, aanmeldmomenten_zv, num_tijdstap))
        ax.plot(x_punten, y_punten, label = zv)
    ax.legend()
    
    # VERLOOP WACHTLIJST grafiek
    fig, ax = plt.subplots()
    ax.set_title('Aantal mensen op wachtlijst')
    ax.set_xlabel('Tijd in weken')
    ax.set_ylabel('Aantal mensen')    
    # Maak gemiddeldelijn over alle simulaties
    x_punten = []
    for x in range(num_tijdstap):
        x_punten.append(x)
    y_punten = []
    for i in range(num_tijdstap):
        y_gem = sum(sim_wachtlijst[:,i])/num_sim
        y_punten.append(y_gem)
    ax.plot(x_punten, y_punten)
    # Onzekerheidsmarge wachtlijst plotten
    y_bovengrens = []
    for i in range(num_tijdstap):
        y_boven = np.quantile(sim_wachtlijst[:,i], 0.85)
        y_bovengrens.append(y_boven) 
    y_ondergrens = []
    for i in range(num_tijdstap):
        y_onder = np.quantile(sim_wachtlijst[:,i],0.15)
        y_ondergrens.append(y_onder)
    ax.fill_between(x_punten, y_ondergrens, y_bovengrens, alpha = 0.5)

    
    # VERLOOP IN_BEHANDELING grafiek
    fig, ax = plt.subplots()
    ax.set_title('Aantal mensen in behandeling')
    ax.set_xlabel('Tijd in weken')
    ax.set_ylabel('Aantal mensen')
    y_punten = []
    for i in range(num_tijdstap):
        y_gem = sum(sim_in_behandeling[:,i])/num_sim
        y_punten.append(y_gem)
    # Gemiddeldelijn plotten
    ax.plot(x_punten, y_punten)
    # Onzekerheidsmarge in_behandeling plotten
    y_bovengrens = []
    for i in range(num_tijdstap):
        y_boven = np.quantile(sim_in_behandeling[:,i], 0.85)
        y_bovengrens.append(y_boven)
    y_ondergrens = []
    for i in range(num_tijdstap):
        y_onder = np.quantile(sim_in_behandeling[:,i], 0.15)
        y_ondergrens.append(y_onder)
    ax.fill_between(x_punten, y_ondergrens, y_bovengrens, alpha = 0.5)
    
    # VERLOOP PLAFONDS grafiek
    fig, ax = plt.subplots()
    ax.set_title('Resterende plekken in budgetplafonds')
    ax.set_xlabel('Tijd in weken')
    ax.set_ylabel('Aantal resterende plekken')
    # Maak voor elke zorgverzekeraar een plot van het gemiddelde
    for k in range(len(zvs)):
        y_punten = []
        for tijdstap in range(num_tijdstap):
            y_gem = sum(sim_plafonds[:, tijdstap, k])/num_sim
            y_punten.append(y_gem)
        ax.plot(x_punten, y_punten, label = zvs[k])
    ax.legend()
    
    
    