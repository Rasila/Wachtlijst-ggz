
"""
@author: Rasila Hoek
"""

# -*- coding: utf-8 -*-
"""
Agent-based model voor wachtlijst ggz, met een aanmeldstop als de wachtlijst
te lang wordt. 

Instroom gemodelleerd met een Poisson-proces 
(parameter: gemidddelde instroom). 
Behandelduur gemodelleerd met een normale verdeling 
(parameters: gemiddelde behandelduur, spreiding behandelduur)
Clienten kunnen drop-out gaan aan het eind van hun wachttijd,
of gedurende de behandeling.
(parameters: kans drop-out wachtlijst, kans drop-out behandeling)

Simuleert hoe de aantallen mensen op de wachtlijst en in behandeling verlopen.
Laat zien hoe de verdeling van wachttijden en het verloop van wachttijd over de tijd 
eruit ziet.   
"""
# ----------------------------------------------------------------------------
# Import modules
import numpy as np
import math
import matplotlib.pyplot as plt

# Helper functions
def flatten(l):
    return [item for sublist in l for item in sublist]
# ----------------------------------------------------------------------------

class client(object):
    
    """
    attributes:
        rest_behandelduur: float, de resterende behandelduur van deze client 
        wachttijd: int, hoe lang heeft deze cliënt tot nu toe gewacht?
    """
    
    def __init__(self, rest_behandelduur, wachttijd):
        # Set attributes
        self.rest_behandelduur = rest_behandelduur
        self.wachttijd = wachttijd
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
        max_wl: int, hoe veel mensen mogen maximaal op wachtlijst
        tijd: int, hoe veel weken loopt de simulatie al
    """
    def __init__(self, wachtlijst, in_behandeling, 
                 max_capaciteit, gem_behandelduur, wl_max, tijd):

        # Set attributes
        self.wachtlijst = wachtlijst
        self.in_behandeling = in_behandeling
        self.max_capaciteit = max_capaciteit
        self.gem_behandelduur = gem_behandelduur
        self.wl_max = wl_max
        self.tijd = tijd
        
    def update(self, instroom, p_dropout_w, p_dropout_b, spreiding_duur):
        """
        Functie die wachtlijst en in_behandeling updatet. 
        Zorgt voor instroom wachtlijst op basis van Poisson met parameter instroom,
            tenzij de wachtlijst vol is.
        Telt hoe veel mensen hadden willen aanmelden tijdens aanmeldstop. 
        Checkt of er behandelplekken vrij zijn en laat wachtenden instromen.
        Checkt of er cliënten klaar zijn om uit te stromen. 
        Checkt of er cliënten drop-out gaan.
        Returns: wachttijden (list), aanmeldmomenten (list) van gestarte clienten,
            en weggestuurd (int), aantal mensen dat tegen aanmeldstop aanliep.
        """
        
        # TIJD BIJHOUDEN
        self.tijd = self.tijd + 1
        
        # INSTROOM WACHTLIJST op basis van Poisson met parameter instroom.
        weggestuurd = 0
        aantal_in = np.random.poisson(instroom)
        if(len(self.wachtlijst) <= self.wl_max):
            # Creëer nieuwe cliënten voor op wachtlijst
            nieuwe_clienten = []
            for i in range(aantal_in):
                # Bepaal behandelduur
                behandelduur = np.random.normal(self.gem_behandelduur, spreiding_duur*self.gem_behandelduur)
                nieuwe_client = client(behandelduur, 0)
                nieuwe_clienten.append(nieuwe_client)
            # Voeg nieuwe clienten toe aan wachtlijst
            self.wachtlijst.extend(nieuwe_clienten)
        else:
            # Registreer dat er mensen niet konden aanmelden
            weggestuurd = aantal_in
        
        # INSTROOM BEHANDELING op basis van plekken vrij en wachtenden
        # Bepaal hoe veel plekken vrij
        plekken_vrij = self.max_capaciteit - len(self.in_behandeling)
        # Initialiseer lijst wachttijden, aanmeldmomenten
        wachttijden = []
        aanmeldmomenten = []
        # Probeer voor elke vrije plek iemand te laten starten
        for i in range(plekken_vrij):
            # Bepaal of er iemand is voor deze plek
            # Zolang er wachtenden zijn...
            while len(self.wachtlijst) > 0:
                # ...neem eerste persoon in overweging als starter
                starter = self.wachtlijst[0]
                # als hij drop_out gaat...
                if(np.random.binomial(1, p_dropout_w)):
                    # ...verwijder van wachtlijst
                    del self.wachtlijst[0]
                # als geen drop-out...
                else:
                    # ..dan gaat deze persoon starten op deze vrije plek (stop while-loop)
                    break
            # Als niemand is gevonden om te starten, stop met proberen (stop for-loop)
            if len(self.wachtlijst) == 0:
                break
            
            # Anders laat persoon starten
            self.in_behandeling.append(starter)
            # Registreer de wachttijd van deze persoon
            wachttijden.append(starter.wachttijd)
            aanmeldmomenten.append(self.tijd - starter.wachttijd)
            # Verwijder deze persoon van wachtlijst
            del self.wachtlijst[0]
        
        # UITSTROOM DOOR AFRONDEN BEHANDELING    
        # Check voor elke client of ie nog resterende behandelduur heeft
        # Zo ja, voeg m toe aan de nieuwe in_behandeling
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
        
        # Return lijsten met wachttijden en aanmeldmomenten
        # En het aantal clienten dat niet kon aanmelden
        return wachttijden, aanmeldmomenten, weggestuurd
    
# ----------------------------------------------------------------------------
def simuleer_wachtlijst_as(num_wachtlijst_start, 
                           rho_start, 
                           max_capaciteit,
                           instroom,
                           gem_behandelduur,
                           spreiding_duur,
                           p_dropout_w,
                           p_dropout_b,
                           max_wl,
                           num_trials,
                           num_tijdstap):
    """
    Functie die de simulatie runt. 
    Args:
        num_wachtlijst_start = int, aantal clienten op wachtlijst bij begin simulatie
        rho_start = float, gedeelte waarvoor de capaciteit is gevuld bij begin simulatie
        max_capaciteit = int, aantal plekken in behandeling
        instroom = float, gemiddelde wekelijkse instroom (aanmeldingen)
        gem_behandelduur = float, gemiddelde behandelduur
        spreiding_duur = float, ratio sd/gemiddelde van behandelduur
        p_dropout_w = kans dat iemand op enig moment tijdens wachtlijst uitvalt
        p_dropout_b = kans dat iemand op enig moment tijdens behandeling uitvalt
        max_wl = int, maximale lengte van de wachtlijst voor aanmeldstop
        num_trials = int, hoe vaak moet de simulatie worden gedaan?
        num_tijdstap = int, hoe veel weken moet de simulatie lopen?
    """
    # Initialiseer matrices/lijsten om resultaten van trials te bewaren
    wachttijden = []
    aanmeldmomenten = []
    sim_wachtlijst = np.zeros((num_trials, num_tijdstap))
    sim_in_behandeling = np.zeros((num_trials, num_tijdstap))
    weggestuurd = []
    
    # Voor num_trials trials
    for trial in range(num_trials):
        
        # Initialiseer lijsten voor wachttijden, aanmeldmomenten in deze trial
        wt_trial = []
        am_trial = []
        weg_trial = 0
        
        # Initialiseer wachtlijst met nieuwe cliënten
        start_wachtlijst = []
        for i in range(num_wachtlijst_start):
            behandelduur = np.random.normal(gem_behandelduur, spreiding_duur*gem_behandelduur)
            start_wachtlijst.append(client(behandelduur, 0))
        
        # Initialiseer in_behandeling met cliënten die al variabel lang in behandeling zijn
        start_in_behandeling = []
        for i in range(math.floor(rho_start*max_capaciteit)):
            behandelduur = np.random.normal(gem_behandelduur, spreiding_duur*gem_behandelduur)
            al_behandeld = np.random.uniform(0, behandelduur)            
            start_in_behandeling.append(client(behandelduur - al_behandeld, 0))
        
        # Initialiseer behandeling met deze wachtlijst, in_behandeling, max_capaciteit
        sim_behandeling = behandeling(start_wachtlijst, start_in_behandeling, 
                                      max_capaciteit, gem_behandelduur, 
                                      max_wl, tijd = 0)
        # Voor elke tijdstap:
        for tijdstap in range(num_tijdstap):
            # Update wachttijd voor alle clienten in wachtlijst
            for clt in sim_behandeling.wachtlijst:
                clt.update_wachttijd()
            # Update rest_behandelduur alle clienten in in_behandeling
            for clt in sim_behandeling.in_behandeling:
                clt.update_behandelduur()
            # Update de behandeling en sla wachttijden en aanmeldmomenten op
            wt, am, weg = sim_behandeling.update(instroom, p_dropout_w, p_dropout_b, spreiding_duur)
            wt_trial.extend(wt)
            am_trial.extend(am)
            weg_trial = weg_trial + weg
            # Bereken en bewaar resultaten
            sim_wachtlijst[trial, tijdstap] = len(sim_behandeling.wachtlijst)
            sim_in_behandeling[trial, tijdstap] = len(sim_behandeling.in_behandeling)
        # Sla wachttijden en aanmeldmomenten van deze trial op
        wachttijden.append(wt_trial)
        aanmeldmomenten.append(am_trial)
        weggestuurd.append(weg_trial)
        
    return sim_wachtlijst, sim_in_behandeling, wachttijden, aanmeldmomenten, weggestuurd, max_capaciteit

# ---------------------------------------------------------------------------------------
def resultaten_simulatie_as(sim_wachtlijst, sim_in_behandeling, wachttijden, 
                         aanmeldmomenten, weggestuurd, max_capaciteit):
    """
    Visualiseert het verloop van de simulatie en geeft samenvatting resultaten. 
    Args:
        sim_wachtlijst: matrix, op positie (i,j) lengte wachtlijst in trial i, tijdstap j.
        sim_in_behandeling: matrix, op positie (i,j) aantal in behandeling in trial i, tijdstap j. 
        wachttijden: List of lists, element [i][j] is wachttijd van j-de starter in trial i. 
        aanmeldmomenten: list of lists, element [i][j] is aanmeldmoment van j-de starter in trial i.
        weggestuurd: list, element i is het aantal mensen dat tegen aanmeldstop aanliep in trial i
        max_capaciteit: aantal behandelplekken
    """
    # Bepaal grootte simulatie
    num_tijdstap = len(sim_wachtlijst[0])
    num_sim = len(sim_wachtlijst)
    num_punten = len(flatten(wachttijden))
    
    # Maak aangepaste lijsten met wachttijden
    # zonder de mensen die al op wachtlijst stonden bij begin simulatie
    # zonder mensen die pas heel laat in de simulatie kwamen
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
    clienten_gem = num_punten/num_sim
    wachttijd_gem_tekst = str(round(wachttijd_gem))
    clienten_gem_tekst = str(round(clienten_gem))
    weg_gem = sum(weggestuurd)/num_sim
    weg_gem_tekst = str(round(weg_gem))
    print(str(num_sim) + " simulaties van " + str(num_tijdstap) + " weken.")
    print("De gemiddelde wachttijd is " + wachttijd_gem_tekst + " weken.")
    print("Er zijn per simulatie gemiddeld " + clienten_gem_tekst + " clienten gestart met behandeling.")
    print("Er zijn per simulatie gemiddeld " + weg_gem_tekst + " mensen tegen een aanmeldstop aangelopen.")
    
    # Bepaal percentage dat korter moest wachten dan treeknorm
    onder_treek = 0
    for trial in range(len(wachttijden_adj)):
        for wachttijd in wachttijden_adj[trial]:
            if(wachttijd < 10):
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
    ax.set_title('Wachttijden per aanmeldmoment (' + str(num_sim) + ' simulaties)')
    ax.set_xlabel('Week van aanmelden')
    ax.set_ylabel('Aantal weken gewacht')
    x_punten = []
    y_punten = []
    aanmeldmomenten_flat = flatten(aanmeldmomenten)
    wachttijden_flat = flatten(wachttijden)
    for i in range(len(aanmeldmomenten_flat)):
        if not(aanmeldmomenten_flat[i] == 0):
            x_punten.append(aanmeldmomenten_flat[i])
            y_punten.append(wachttijden_flat[i])
    ax.scatter(x_punten, y_punten, s = 1)
    # Plot de lijnen voor treeknorm en gemiddelde en maak een legenda
    ax.axhline(y=14, color='r', linestyle='--', linewidth = 1, label = 'treeknorm')
    ax.axhline(y=wachttijd_gem, color='black', linestyle = '--', linewidth = 1, label = 'gemiddelde')
    ax.legend()
    # Plot eindlijn
    x_eindlijn = np.linspace(num_tijdstap - max(wachttijden_flat), num_tijdstap, 100)
    y_eindlijn = num_tijdstap - x_eindlijn
    ax.fill_between(x_eindlijn, y_eindlijn, max(wachttijden_flat), alpha = 0.5, color = 'black')

    
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
    x_punten = []
    for x in range(num_tijdstap):
        x_punten.append(x)
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
    
sim_w, sim_b, wt, am, weg, mc = simuleer_wachtlijst_as(
                    num_wachtlijst_start = 2, 
                    rho_start = 1, 
                    max_capaciteit = 15,
                    instroom = 15/78,
                    gem_behandelduur = 78,
                    spreiding_duur = 0.2,
                    p_dropout_w = 0.1,
                    p_dropout_b = 0.1,
                    max_wl = 3,
                    num_trials = 100,
                    num_tijdstap = 260)
resultaten_simulatie_as(sim_w, sim_b, wt, am, weg, mc)