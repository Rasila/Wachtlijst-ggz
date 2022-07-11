# -*- coding: utf-8 -*-
"""
Agent-based model voor wachtlijst ggz. 

Instroom gemodelleerd met een Poisson-proces 
(parameter: gemidddelde instroom). 
Behandelduur gemodelleerd met een normale verdeling 
(parameters: gemiddelde behandelduur, spreiding behandelduur)
Clienten kunnen drop-out gaan aan het eind van hun wachttijd,
of gedurende de behandeling.
(parameters: kans drop-out wachtlijst, kans drop-out behandeling)

Brengt in beeld hoe de aantallen mensen op de wachtlijst en in behandeling verlopen.
Laat zien hoe de verdeling van wachttijden en het verloop van wachttijd over de tijd 
eruit ziet.   
"""

# Import modules
import numpy as np
import math
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
class client(object):
    
    """
    attributes:
        rest_behandelduur: float, de behandelduur van deze client 
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
    behandeling
    attributes:
        wachtlijst: list, cliënten op de wachtlijst
        in_behandeling: list, cliënten in behandeling
        max_capaciteit: int, totaal aantal behandelplekken
        gem_behandelduur: int, gemiddelde behandelduur in weken
        tijd: int, hoe veel weken loopt de simulatie al
        
    """
    def __init__(self, wachtlijst, in_behandeling, 
                 max_capaciteit, gem_behandelduur, tijd):

        # Set attributes
        self.wachtlijst = wachtlijst
        self.in_behandeling = in_behandeling
        self.max_capaciteit = max_capaciteit
        self.gem_behandelduur = gem_behandelduur
        self.tijd = tijd
        
    def update(self, instroom, p_dropout_w, p_dropout_b, spreiding_duur):
        """
        Functie die wachtlijst en in_behandeling updatet. 
        Zorgt voor instroom wachtlijst op basis van Poisson met parameter instroom.
        Checkt of er behandelplekken vrij zijn en laat wachtenden instromen.
        Checkt of er cliënten klaar zijn om uit te stromen. 
        Checkt of er cliënten drop-out gaan.
        Returns: wachttijden en aanmeldmomenten van in deze tijdstap gestarte clienten. 
        """
        
        # TIJD BIJHOUDEN
        self.tijd = self.tijd + 1
        
        # INSTROOM WACHTLIJST op basis van Poisson met parameter instroom.
        aantal_in = np.random.poisson(instroom)
        # Creëer nieuwe cliënten voor op wachtlijst
        nieuwe_clienten = []
        for i in range(aantal_in):
            # Bepaal behandelduur (Hier spreiding behandelduur aannemen)
            behandelduur = np.random.normal(self.gem_behandelduur, spreiding_duur*self.gem_behandelduur)
            nieuwe_client = client(behandelduur, 0)
            nieuwe_clienten.append(nieuwe_client)
        # Voeg nieuwe clienten toe aan wachtlijst
        self.wachtlijst.extend(nieuwe_clienten)
        
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
                    # ..is er een persoon die gaat starten op deze vrije plek (stop while-loop)
                    break
            # Als niemand is gevonden om te starten, stop met proberen (stop for-loop)
            if len(self.wachtlijst) == 0:
                break
            
            # Anders gaat persoon starten
            self.in_behandeling.append(starter)
            # Registreer de wachttijd van deze persoon
            wachttijden.append(starter.wachttijd)
            aanmeldmomenten.append(self.tijd - starter.wachttijd)
            # Verwijder deze persoon van wachtlijst
            del self.wachtlijst[0]
        
        # UITSTROOM DOOR AFRONDEN BEHANDELING    
        # Check voor elke client of ie nog in behandeling blijft
        # Zo niet, voeg m niet toe aan de nieuwe in_behandeling
        nieuwe_in_behandeling = []
        for clt in self.in_behandeling:
           if not(clt.rest_behandelduur < 0):
               nieuwe_in_behandeling.append(clt)
        self.in_behandeling = nieuwe_in_behandeling
        
        # DROP-OUT BEHANDELING
        # Voor elke in behandeling, bepaal wel of geen drop-out
        in_behandeling_gedropt = []
        for clt in self.in_behandeling:
            if not(np.random.binomial(1, 1-(1-p_dropout_b)**(1/self.gem_behandelduur))):
                in_behandeling_gedropt.append(clt)
        self.in_behandeling = in_behandeling_gedropt
        
        # Return lijsten met wachttijden en aanmeldmomenten
        return wachttijden, aanmeldmomenten
    
# ----------------------------------------------------------------------------
def simuleer_wachtlijst(num_wachtlijst_start, 
                        rho_start, 
                        max_capaciteit,
                        instroom,
                        gem_behandelduur,
                        spreiding_duur,
                        p_dropout_w,
                        p_dropout_b,
                        num_trials,
                        num_tijdstap):
    """
    Functie die de simulatie runt. 
    Args:
        num_wachtlijst_start = int, aantal clienten op wachtlijst bij begin simulatie
        rho_start = float, gedeelte waarvoor de capaciteit is gevuld bij begin simulatie
        max_capaciteit = int, aantal plekken in behandeling
        instroom = float, gemiddelde wekelijkse instroom
        gem_behandelduur = float, gemiddelde behandelduur
        spreiding_duur = float, ratio spreiding/gemiddelde van behandelduur
        p_dropout_w = kans dat iemand ergens tijdens wachtlijst uitvalt
        p_dropout_b = kans dat iemand ergens tijdens behandeling uitvalt
    """
    # Initialiseer matrices/lijsten om resultaten van trials te bewaren
    wachttijden = []
    aanmeldmomenten = []
    sim_wachtlijst = np.zeros((num_trials, num_tijdstap))
    sim_in_behandeling = np.zeros((num_trials, num_tijdstap))
    
    # Voor num_trials trials
    for trial in range(num_trials):
        
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
                                      max_capaciteit, gem_behandelduur, tijd = 0)
        # Voor elke tijdstap:
        for tijdstap in range(num_tijdstap):
            # Update wachttijd voor alle clienten in wachtlijst
            for clt in sim_behandeling.wachtlijst:
                clt.update_wachttijd()
            # Update rest_behandelduur alle clienten in in_behandeling
            for clt in sim_behandeling.in_behandeling:
                clt.update_behandelduur()
            # Update de behandeling en sla wachttijden en aanmeldmomenten op
            wt, am = sim_behandeling.update(instroom, p_dropout_w, p_dropout_b, spreiding_duur)
            wachttijden.extend(wt)
            aanmeldmomenten.extend(am)
            # Bereken en bewaar resultaten
            sim_wachtlijst[trial, tijdstap] = len(sim_behandeling.wachtlijst)
            sim_in_behandeling[trial, tijdstap] = len(sim_behandeling.in_behandeling)
            rho = sim_in_behandeling/max_capaciteit
            
    return sim_wachtlijst, sim_in_behandeling, wachttijden, aanmeldmomenten, rho, max_capaciteit
        
# -------------------------------------------------------------------------------

def gemiddelde_wachttijd(t, delta, wachttijden, aanmeldmomenten, num_tijdstap):
    """
    Hulpfunctie voor resultaten_simulatie()
    Bepaalt gemiddelde wachttijd rond bepaald tijdstip over meerdere simulaties.
    
    t = int, aanmeldmoment waarvoor wachttijd bepaald moet worden
    delta = int, lengte van het tijdvak
    wachttijden = list, lijst met wachttijden uit simulatie
    aanmeldmomenten = list, lijst met bijbehorende aanmeldmomenten uit simulatie 
    num_tijdstap = int, hoe lang liep de simulatie?
    
    returns: wachttijd
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

def resultaten_simulatie(sim_wachtlijst, sim_in_behandeling, wachttijden, aanmeldmomenten, 
                         rho, max_capaciteit):
    """
    Visualiseert het verloop van de simulatie en geeft samenvatting resultaten. 
    sim_wachtlijst: Matrix met op positie (i,j) de wachtlijst in trial i, tijdstap j.
    sim_in_behandeling: Matrix met op positie (i,j) de wachtlijst in trial i, tijdstap j. 
    wachttijden: Lijst met wachttijden van alle starters tijdens alle trials. 
    aanmeldmomenten: Lijst met aanmeldmomenten van alle starters tijdens alle trials
    rho: Matrix met op positie (i,j) de mate waarin behandeling gevuld is in trial i, tijdstap j. 
    max_capaciteit: Aantal plekken in behandeling
    """
    # Bepaal grootte simulatie
    num_tijdstap = len(sim_wachtlijst[0])
    num_sim = len(sim_wachtlijst)
    num_punten = len(wachttijden)
    
    # Maak aangepaste lijst met wachttijden, alleen unbiased data en zonder 
    # de mensen die al op wachtlijst stonden bij begin simulatie
    wachttijden_adj = []
    for i in range(len(wachttijden)):
        if(aanmeldmomenten[i] < num_tijdstap - max(wachttijden)):
            if not(aanmeldmomenten[i] == 0):
                wachttijden_adj.append(wachttijden[i])
    
    # Enkele resultaten om te printen
    wachttijd_gem = sum(wachttijden_adj)/len(wachttijden_adj)
    clienten_gem = num_punten/num_sim
    rho_gem = sum(sum(rho))/np.size(rho)
    wachttijd_gem_tekst = str(round(wachttijd_gem))
    rho_gem_tekst = str(round(100*rho_gem))
    clienten_gem_tekst = str(round(clienten_gem))
    print(str(num_sim) + " simulaties van " + str(num_tijdstap) + " weken.")
    print("De gemiddelde wachttijd is " + wachttijd_gem_tekst + " weken.")
    print("De capaciteit is gemiddeld voor " + rho_gem_tekst + " procent gevuld.")
    print("Er zijn per simulatie gemiddeld " + clienten_gem_tekst + " clienten gestart met behandeling.")
    # Bepaal percentage dat korter moest wachten dan treeknorm
    onder_treek = 0
    for wachttijd in wachttijden_adj:
        if(wachttijd < 14):
            onder_treek = onder_treek + 1
    procent_onder_treek = round(100*onder_treek/len(wachttijden_adj))
    print(str(procent_onder_treek) + " procent is binnen treeknorm gestart.")
    
    # VERDELING WACHTTIJDEN
    fig, ax = plt.subplots()
    ax.set_title('Verdeling wachttijd in ' + str(num_sim) + ' simulaties')
    ax.hist(wachttijden_adj)
    ax.set_xlabel('Aantal weken')
    
    # SCATTERPLOT WACHTTIJD PER AANMELDMOMENT
    fig, ax = plt.subplots()
    ax.set_title('Wachttijden per aanmeldmoment (' + str(num_sim) + ' simulaties)')
    ax.set_xlabel('Week van aanmelden')
    ax.set_ylabel('Aantal weken gewacht')
    x_punten = []
    y_punten = []
    for i in range(len(aanmeldmomenten)):
        if not(aanmeldmomenten[i] == 0):
            x_punten.append(aanmeldmomenten[i])
            y_punten.append(wachttijden[i])
    ax.scatter(x_punten, y_punten, s = 1)
    # Plot de lijnen voor treeknorm en gemiddelde en maak een legenda
    ax.axhline(y=14, color='r', linestyle='--', linewidth = 1, label = 'treeknorm')
    ax.axhline(y=wachttijd_gem, color='black', linestyle = '--', linewidth = 1, label = 'gemiddelde')
    ax.legend()
    # Plot eindlijn
    x_eindlijn = np.linspace(num_tijdstap - max(wachttijden), num_tijdstap, 100)
    y_eindlijn = num_tijdstap - x_eindlijn
    ax.fill_between(x_eindlijn, y_eindlijn, max(wachttijden), alpha = 0.5, color = 'black')
    
    # VERLOOP GEMIDDELDE WACHTTIJD 
    if not(num_sim == 1):
        fig, ax = plt.subplots()
        ax.set_title('Verloop gemiddelde wachttijd')
        ax.set_xlabel('Week van aanmelden')
        ax.set_ylabel('Wachttijd')
        # Plot de lijn van de gemiddelde wachttijd
        delta = int(num_tijdstap/13)
        x_punten = []
        for x in range(int(num_tijdstap - max(wachttijden) - 0.5*delta)):
            x_punten.append(x)
        y_punten = []
        for x in x_punten:
            y_punten.append(gemiddelde_wachttijd(
                x, delta, wachttijden, aanmeldmomenten, num_tijdstap))
        ax.plot(x_punten, y_punten)
    
    # VERLOOP WACHTLIJST
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

    
    # VERLOOP IN_BEHANDELING
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

#------------------------------------------------------------------------------

# TEST Simulatie
sim_w, sim_ib, wt, am, rho, max_capaciteit = simuleer_wachtlijst(
                    num_wachtlijst_start = 100,
                    rho_start = 1, 
                    max_capaciteit = 40, 
                    instroom = 35/26, 
                    gem_behandelduur = 26,
                    spreiding_duur = 0.2,
                    p_dropout_w = 0.05,
                    p_dropout_b = 0.05,
                    num_trials = 100, 
                    num_tijdstap = 100) 
resultaten_simulatie(sim_w, sim_ib, wt, am, rho, max_capaciteit)

# -----------------------------------------------------------------------------

    
    